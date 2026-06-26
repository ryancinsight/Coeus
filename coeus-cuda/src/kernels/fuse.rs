use super::GpuLayoutInfo;
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::{get_cuda_context, CUdeviceptr, CUfunction, CUmodule, CudaDriver, NvrtcDriver};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
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

pub fn compile_cuda_to_ptx(src: &str) -> Result<String, String> {
    let nvrtc = NvrtcDriver::get().ok_or_else(|| "NVRTC driver not available".to_string())?;

    let src_c = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new("fused_kernel.cu").map_err(|e| e.to_string())?;

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

        let options = [std::ffi::CString::new("--std=c++11").unwrap()];
        let options_ptr: Vec<*const std::ffi::c_char> =
            options.iter().map(|o| o.as_ptr()).collect();

        let compile_res = (nvrtc.nvrtcCompileProgram)(
            prog,
            options_ptr.len() as std::ffi::c_int,
            options_ptr.as_ptr(),
        );

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
        // `CString::new(ptx)` downstream fails, silently degrading all JIT
        // kernels to the CPU fallback.
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
        let func_name_c = std::ffi::CString::new(func_name).unwrap();
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
    let cuda_type = T::CUDA_TYPE;

    // 1. Collect unique input tensors
    let mut input_ptrs = Vec::new();
    expr.collect_inputs(&mut input_ptrs);
    let num_inputs = input_ptrs.len();

    let inputs: Vec<&Tensor<T, CudaBackend>> = input_ptrs.iter().map(|&p| unsafe { &*p }).collect();

    // 2. Build input pointer to index map
    let mut input_map = HashMap::new();
    for (i, &p) in input_ptrs.iter().enumerate() {
        input_map.insert(p, i);
    }

    // 3. Generate the shader expression string
    let expr_str = expr.to_shader_expr(&input_map);

    // 4. Create layouts buffer
    let mut layouts_gpu = Vec::with_capacity(num_inputs + 1);
    for input in &inputs {
        layouts_gpu.push(GpuLayoutInfo::from_layout(input.layout()));
    }
    layouts_gpu.push(GpuLayoutInfo::from_layout(out_layout));

    let size_u32 = layouts_gpu.len() * (std::mem::size_of::<GpuLayoutInfo>() / 4);
    let mut layout_buf = CudaStorage::<u32>::new(size_u32);
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
    let mut n_val = out_layout.numel() as u32;

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

    let block_size = 256;
    let grid_size = out_layout.numel().div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32,
            1,
            1,
            block_size as u32,
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
