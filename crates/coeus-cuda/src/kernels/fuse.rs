use super::GpuLayoutInfo;
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::{CUdeviceptr, CUfunction, CUmodule, CudaDriver, NvrtcDriver, get_cuda_context};
use crate::kernels::validation::{
    CUDA_BLOCK_SIZE, checked_layout_storage_len, checked_numel, cuda_u32, launch_grid_size,
    layouts_fit_cuda,
};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use coeus_tensor::broadcast::broadcast_shapes;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

pub(crate) struct SafeCachedKernel {
    pub(crate) module: CUmodule,
    pub(crate) func: CUfunction,
}

unsafe impl Send for SafeCachedKernel {}
unsafe impl Sync for SafeCachedKernel {}

impl Drop for SafeCachedKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Some(drv) = CudaDriver::get() {
                unsafe {
                    (drv.cu_module_unload)(self.module);
                }
            }
        }
    }
}

// Internal `KERNEL_CACHE` uses `RwLock<HashMap<…>>` instead of `Mutex<HashMap<…>>`
// so concurrent cache-hit lookups on the read path do not serialize through a
// single mutex. Cache hits are statistically dominant once fused expressions
// reach steady state, so the read-mostly workload benefits directly from
// concurrent read-locks; cache misses amortize the rare write-lock acquisition
// over the subsequent `cu_module_load_data` cost. Strict monotonic improvement:
// no API change observable to callers; `Arc::clone()` returns the same shared
// kernel reference regardless of lock type.
static KERNEL_CACHE: OnceLock<RwLock<HashMap<String, Arc<SafeCachedKernel>>>> = OnceLock::new();

/// Compile a CUDA C source string to PTX using the NVRTC driver.
///
/// Returns the PTX assembly string on success or an error message on failure.
pub fn compile_cuda_to_ptx(src: &str) -> Result<String, String> {
    let nvrtc = NvrtcDriver::get().ok_or_else(|| "NVRTC driver not available".to_string())?;

    let src_c = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new("fused_kernel.cu").map_err(|e| e.to_string())?;
    let mut option_text = vec!["--std=c++11".to_string()];
    if let Some(toolkit) =
        std::env::var_os("CUDA_TOOLKIT_PATH").or_else(|| std::env::var_os("CUDA_PATH"))
    {
        option_text.push(format!(
            "--include-path={}",
            std::path::Path::new(&toolkit).join("include").display()
        ));
    }
    let options: Result<Vec<_>, _> = option_text
        .iter()
        .map(|option| std::ffi::CString::new(option.as_str()))
        .collect();
    let options = options.map_err(|error| format!("invalid NVRTC option: {error}"))?;
    let options_ptr: Vec<*const std::ffi::c_char> =
        options.iter().map(|option| option.as_ptr()).collect();
    let option_count = std::ffi::c_int::try_from(options_ptr.len())
        .map_err(|_| "too many NVRTC compiler options".to_string())?;

    let mut prog: crate::driver::nvrtcProgram = std::ptr::null_mut();
    unsafe {
        let res = (nvrtc.nvrtcCreateProgram)(
            &mut prog,
            src_c.as_ptr(),
            name_c.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if res != 0 {
            return Err(format!("nvrtcCreateProgram failed: {}", res));
        }

        let compile_res = (nvrtc.nvrtcCompileProgram)(prog, option_count, options_ptr.as_ptr());

        if compile_res != 0 {
            let mut log_size: usize = 0;
            (nvrtc.nvrtcGetProgramLogSize)(prog, &mut log_size);
            let mut log_bytes = vec![0u8; log_size];
            (nvrtc.nvrtcGetProgramLog)(prog, log_bytes.as_mut_ptr() as *mut std::ffi::c_char);
            let log_str = String::from_utf8_lossy(&log_bytes).into_owned();

            (nvrtc.nvrtcDestroyProgram)(&mut prog);
            return Err(format!(
                "nvrtcCompileProgram failed (code {}). Log:\n{}",
                compile_res, log_str
            ));
        }

        let mut ptx_size: usize = 0;
        let ptx_res = (nvrtc.nvrtcGetPTXSize)(prog, &mut ptx_size);
        if ptx_res != 0 {
            (nvrtc.nvrtcDestroyProgram)(&mut prog);
            return Err(format!("nvrtcGetPTXSize failed: {}", ptx_res));
        }

        let mut ptx_bytes = vec![0u8; ptx_size];
        let ptx_get_res =
            (nvrtc.nvrtcGetPTX)(prog, ptx_bytes.as_mut_ptr() as *mut std::ffi::c_char);
        if ptx_get_res != 0 {
            (nvrtc.nvrtcDestroyProgram)(&mut prog);
            return Err(format!("nvrtcGetPTX failed: {}", ptx_get_res));
        }

        (nvrtc.nvrtcDestroyProgram)(&mut prog);

        // `nvrtcGetPTXSize` reports the buffer size *including* the trailing NUL
        // terminator, so `ptx_bytes` ends in one or more NUL bytes. Trim them;
        // otherwise the PTX `String` carries an interior NUL and every
        // `CString::new(ptx)` downstream fails, making all JIT kernels
        // unavailable at the CUDA dispatch boundary.
        while ptx_bytes.last() == Some(&0) {
            ptx_bytes.pop();
        }

        let ptx_str =
            String::from_utf8(ptx_bytes).map_err(|e| format!("PTX is not valid UTF-8: {}", e))?;
        Ok(ptx_str)
    }
}

pub(crate) fn get_or_create_kernel(
    expr_str: &str,
    cuda_src: &str,
    func_name: &str,
) -> Option<Arc<SafeCachedKernel>> {
    let cache = KERNEL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    // Fast path: read-lock. Concurrent cache hits load the same `Arc<SafeCachedKernel>`
    // reference without serialising through a single critical section.
    {
        let map = cache.read().ok()?;
        if let Some(kernel) = map.get(expr_str) {
            return Some(kernel.clone());
        }
    }

    let ptx = compile_cuda_to_ptx(cuda_src).ok()?;
    let drv = CudaDriver::get()?;
    let ptx_c = std::ffi::CString::new(ptx).ok()?;
    let mut module = std::ptr::null_mut();

    let kernel = unsafe {
        let res = (drv.cu_module_load_data)(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void);
        if res != 0 {
            return None;
        }

        let mut func = std::ptr::null_mut();
        let func_name_c = std::ffi::CString::new(func_name).ok()?;
        let res = (drv.cu_module_get_function)(&mut func, module, func_name_c.as_ptr());
        if res != 0 {
            (drv.cu_module_unload)(module);
            return None;
        }

        Arc::new(SafeCachedKernel { module, func })
    };

    // Slow path: write-lock on cache insertion. Re-check for an entry inserted by
    // a concurrent thread between our read-lock and write-lock acquisitions so the
    // last writer wins on duplicates rather than leaking the unloaded module.
    let mut map = cache.write().ok()?;
    if let Some(existing) = map.get(expr_str) {
        return Some(existing.clone());
    }
    map.insert(expr_str.to_string(), kernel.clone());
    Some(kernel)
}

/// Compile and dispatch a dynamically generated fused CUDA kernel.
pub fn dispatch_fused<T: CudaScalar, E: ExprNode<T, CudaBackend>>(
    expr: &E,
    output: &mut CudaStorage<T>,
    out_layout: &Layout,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    let Some(total) = checked_numel(out_layout) else {
        return false;
    };
    let Some(grid_size) = launch_grid_size(total) else {
        return false;
    };
    if out_layout.offset() != 0 || !out_layout.is_contiguous() || output.len() < total {
        return false;
    }
    let cuda_type = T::CUDA_TYPE;

    // 1. Collect unique input tensors
    let mut input_ptrs = Vec::new();
    expr.collect_inputs(&mut input_ptrs);
    let num_inputs = input_ptrs.len();
    let Some(layout_count) = num_inputs.checked_add(1) else {
        return false;
    };
    if input_ptrs.iter().any(|ptr| ptr.is_null()) {
        return false;
    }

    let inputs: Vec<&Tensor<T, CudaBackend>> = input_ptrs
        .iter()
        .map(|&p| {
            // SAFETY: ExprNode input collection returns pointers to the tensors
            // captured by the expression; null pointers were rejected above.
            unsafe { &*p }
        })
        .collect();

    for input in &inputs {
        let input_layout = input.layout();
        let Some(broadcast_shape) = broadcast_shapes(input_layout.shape(), out_layout.shape())
        else {
            return false;
        };
        let Some(required) = checked_layout_storage_len(input_layout) else {
            return false;
        };
        if broadcast_shape.as_ref() != out_layout.shape()
            || !layouts_fit_cuda(&[input_layout])
            || required.checked_sub(1).and_then(cuda_u32).is_none()
            || input.storage().len() < required
        {
            return false;
        }
    }

    // 2. Build input pointer to index map
    let mut input_map = HashMap::new();
    for (i, &p) in input_ptrs.iter().enumerate() {
        input_map.insert(p, i);
    }

    // 3. Generate the shader expression string
    let expr_str = expr.to_shader_expr(&input_map);

    // 4. Create layouts buffer
    let mut layouts_gpu = Vec::new();
    if layouts_gpu.try_reserve_exact(layout_count).is_err() {
        return false;
    }
    for input in &inputs {
        let Ok(layout) = GpuLayoutInfo::try_from(input.layout()) else {
            return false;
        };
        layouts_gpu.push(layout);
    }
    let Ok(out_layout_gpu) = GpuLayoutInfo::try_from(out_layout) else {
        return false;
    };
    layouts_gpu.push(out_layout_gpu);

    let Some(size_u32) = layouts_gpu
        .len()
        .checked_mul(std::mem::size_of::<GpuLayoutInfo>() / 4)
    else {
        return false;
    };
    let mut layout_buf = CudaStorage::<u32>::new(size_u32);
    // SAFETY: `GpuLayoutInfo` is `#[repr(C)]` and derives `bytemuck::Pod`, so
    // its initialized contiguous storage can be viewed as `u32` words.
    let slice = unsafe { std::slice::from_raw_parts(layouts_gpu.as_ptr() as *const u32, size_u32) };
    CudaBackend::new().copy_to_device(slice, &mut layout_buf);

    // 5. Generate C++ CUDA kernel source code
    let mut offset_calcs = String::new();
    for i in 0..num_inputs {
        offset_calcs.push_str(&format!(
            "    unsigned int off_{0} = layout_infos[{0}].offset;\n\
             for (unsigned int d = 0; d < out_layout.ndim; ++d) {{\n\
                 if (d >= out_layout.ndim - layout_infos[{0}].ndim) {{\n\
                     unsigned int ad = d + layout_infos[{0}].ndim - out_layout.ndim;\n\
                     if (layout_infos[{0}].shape[ad] > 1) {{\n\
                         off_{0} += coords[d] * layout_infos[{0}].strides[ad];\n\
                     }}\n\
                 }}\n\
             }}\n\
             {1} val_{0} = t_{0}[off_{0}];\n\n",
            i, cuda_type
        ));
    }

    let mut params = Vec::new();
    params.push(format!("{}* out", cuda_type));
    params.push("const GpuLayoutInfo* layout_infos".to_string());
    params.push("unsigned int n".to_string());
    for i in 0..num_inputs {
        params.push(format!("const {}* t_{}", cuda_type, i));
    }
    let params_str = params.join(", ");

    let cuda_src = format!(
        r#"
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

struct GpuLayoutInfo {{
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
}};

extern "C" __global__ void fused_kernel(
    {params_str}
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {{
        return;
    }}

    GpuLayoutInfo out_layout = layout_infos[{binding_out}];
    unsigned int temp = idx;
    unsigned int coords[8] = {{0}};
    for (unsigned int d = 0; d < out_layout.ndim; ++d) {{
        coords[d] = temp / out_layout.strides[d];
        temp = temp % out_layout.strides[d];
    }}

{offset_calcs}
    out[idx] = {expr_str};
}}
"#,
        params_str = params_str,
        binding_out = num_inputs,
        offset_calcs = offset_calcs,
        expr_str = expr_str
    );

    // 6. Get or create kernel module
    let key = format!("fused_{}_{}", expr_str, cuda_type);
    let Some(kernel) = get_or_create_kernel(&key, &cuda_src, "fused_kernel") else {
        return false;
    };

    // 7. Marshal arguments and launch
    let mut out_ptr = output.cu_deviceptr();
    let mut layouts_ptr = layout_buf.cu_deviceptr();
    let Some(mut n_val) = cuda_u32(total) else {
        return false;
    };

    let mut in_ptrs: Vec<CUdeviceptr> = inputs
        .iter()
        .map(|input| input.storage().cu_deviceptr())
        .collect();

    let mut args = Vec::new();
    args.push(&mut out_ptr as *mut CUdeviceptr as *mut std::ffi::c_void);
    args.push(&mut layouts_ptr as *mut CUdeviceptr as *mut std::ffi::c_void);
    args.push(&mut n_val as *mut u32 as *mut std::ffi::c_void);
    for ptr in &mut in_ptrs {
        args.push(ptr as *mut CUdeviceptr as *mut std::ffi::c_void);
    }

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size,
            1,
            1,
            CUDA_BLOCK_SIZE,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}
