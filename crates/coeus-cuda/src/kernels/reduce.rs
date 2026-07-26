use super::fuse::get_or_create_kernel;
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::{get_cuda_context, CUdeviceptr, CudaDriver};
use crate::kernels::validation::{
    checked_numel, cuda_u32, launch_grid_size, layouts_fit_cuda, CUDA_BLOCK_SIZE,
};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use std::collections::HashMap;

/// Compile and dispatch a dynamically generated CUDA reduction kernel for Sum, Max, and Min.
pub fn dispatch_reduce<T: CudaScalar>(
    op: coeus_ops::ReductionOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    if a_layout.shape().get(axis).is_none() || !layouts_fit_cuda(&[a_layout, c_layout]) {
        return false;
    }
    let Some(out_numel) = checked_numel(c_layout) else {
        return false;
    };
    let Some(axis_value) = cuda_u32(axis) else {
        return false;
    };
    let Some(out_numel_value) = cuda_u32(out_numel) else {
        return false;
    };
    let Some(grid_size) = launch_grid_size(out_numel) else {
        return false;
    };
    let Ok(gpu_a_layout) = crate::kernels::GpuLayoutInfo::try_from(a_layout) else {
        return false;
    };
    let Ok(gpu_c_layout) = crate::kernels::GpuLayoutInfo::try_from(c_layout) else {
        return false;
    };
    let cuda_type = T::CUDA_TYPE;

    let (init_expr, loop_start, update_expr) = match op {
        coeus_ops::ReductionOp::Sum => ("0.0f", "0", "acc = acc + val;"),
        coeus_ops::ReductionOp::Prod => ("1.0f", "0", "acc = acc * val;"),
        coeus_ops::ReductionOp::Mean => ("0.0f", "0", "acc = acc + val;"),
        coeus_ops::ReductionOp::Max => ("a[base_off_a]", "1", "acc = max(acc, val);"),
        coeus_ops::ReductionOp::Min => ("a[base_off_a]", "1", "acc = min(acc, val);"),
    };
    let final_expr = match op {
        coeus_ops::ReductionOp::Mean => format!("acc / static_cast<{cuda_type}>(axis_len)"),
        _ => "acc".to_string(),
    };

    let cuda_src = format!(
        r#"
struct GpuLayoutInfo {{
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
}};

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

extern "C" __global__ void reduce_kernel(
    const {cuda_type}* a,
    {cuda_type}* c,
    GpuLayoutInfo a_layout,
    GpuLayoutInfo c_layout,
    unsigned int axis,
    unsigned int out_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) {{
        return;
    }}
    
    unsigned int axis_len = a_layout.shape[axis];
    unsigned int temp = idx;
    unsigned int base_off_a = a_layout.offset;
    
    for (unsigned int d = 0; d < a_layout.ndim; ++d) {{
        unsigned int coord = temp / c_layout.strides[d];
        temp = temp % c_layout.strides[d];
        if (d != axis) {{
            base_off_a += coord * a_layout.strides[d];
        }}
    }}
    
    {cuda_type} acc = {init_expr};
    unsigned int stride_axis = a_layout.strides[axis];
    if (axis_len > 0) {{
        for (unsigned int k = {loop_start}; k < axis_len; ++k) {{
            {cuda_type} val = a[base_off_a + k * stride_axis];
            {update_expr}
        }}
    }}
    
    c[idx] = {final_expr};
}}
"#,
        cuda_type = cuda_type,
        init_expr = init_expr,
        loop_start = loop_start,
        update_expr = update_expr,
        final_expr = final_expr
    );

    let key = format!("reduce_val_{:?}_{}", op, cuda_type);
    let Some(kernel) = get_or_create_kernel(&key, &cuda_src, "reduce_kernel") else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut axis_val = axis_value;
    let mut out_n = out_numel_value;

    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_a_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &gpu_c_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &mut axis_val as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];

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

/// Compile and dispatch a dynamically generated CUDA reduction kernel for fused element-wise and reduction.
pub fn dispatch_fused_reduce<T: CudaScalar, E: ExprNode<T, CudaBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    let Some(expr_shape) = expr.shape() else {
        return false;
    };
    let expr_ndim = expr_shape.len();
    if expr_ndim > 8 {
        return false;
    }
    let Some(&axis_len) = expr_shape.get(axis) else {
        return false;
    };
    let Some(axis_value) = cuda_u32(axis) else {
        return false;
    };
    let Some(axis_len_value) = cuda_u32(axis_len) else {
        return false;
    };
    if !layouts_fit_cuda(&[c_layout]) {
        return false;
    }
    let Some(out_numel) = checked_numel(c_layout) else {
        return false;
    };
    let Some(out_numel_value) = cuda_u32(out_numel) else {
        return false;
    };
    let Some(grid_size) = launch_grid_size(out_numel) else {
        return false;
    };
    let Ok(c_layout_gpu) = crate::kernels::GpuLayoutInfo::try_from(c_layout) else {
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
        if !layouts_fit_cuda(&[input.layout()]) {
            return false;
        }
        let Ok(layout) = crate::kernels::GpuLayoutInfo::try_from(input.layout()) else {
            return false;
        };
        layouts_gpu.push(layout);
    }
    // We add c_layout as the last one to decode output coordinates
    layouts_gpu.push(c_layout_gpu);

    let slice: &[u32] = bytemuck::cast_slice(&layouts_gpu);
    let size_u32 = slice.len();
    let mut layout_buf = CudaStorage::<u32>::new(size_u32);
    CudaBackend::new().copy_to_device(slice, &mut layout_buf);

    // 5. Generate C++ CUDA kernel source code
    let mut offset_calcs = String::new();
    for i in 0..num_inputs {
        offset_calcs.push_str(&format!(
            "        unsigned int off_{0} = layout_infos[{0}].offset;\n\
             for (unsigned int d = 0; d < {1}; ++d) {{\n\
                  if (d >= {1} - layout_infos[{0}].ndim) {{\n\
                      unsigned int ad = d + layout_infos[{0}].ndim - {1};\n\
                      if (layout_infos[{0}].shape[ad] > 1) {{\n\
                          off_{0} += coords[d] * layout_infos[{0}].strides[ad];\n\
                      }}\n\
                  }}\n\
              }}\n\
              {2} val_{0} = t_{0}[off_{0}];\n\n",
            i, expr_ndim, cuda_type
        ));
    }

    let mut params = vec![
        format!("{}* out", cuda_type),
        "const GpuLayoutInfo* layout_infos".to_string(),
        "unsigned int axis".to_string(),
        "unsigned int axis_len".to_string(),
        "unsigned int out_numel".to_string(),
    ];
    for i in 0..num_inputs {
        params.push(format!("const {}* t_{}", cuda_type, i));
    }
    let params_str = params.join(", ");

    let (init_val, update_expr) = match op {
        coeus_ops::ReductionOp::Sum => ("0.0f", "acc = acc + val;"),
        coeus_ops::ReductionOp::Prod => ("1.0f", "acc = acc * val;"),
        coeus_ops::ReductionOp::Mean => ("0.0f", "acc = acc + val;"),
        coeus_ops::ReductionOp::Max => ("-3.40282347e+38f", "acc = max(acc, val);"),
        coeus_ops::ReductionOp::Min => ("3.40282347e+38f", "acc = min(acc, val);"),
    };
    let final_expr = match op {
        coeus_ops::ReductionOp::Mean => format!("acc / static_cast<{cuda_type}>(axis_len)"),
        _ => "acc".to_string(),
    };

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

extern "C" __global__ void fused_reduce_kernel(
    {params_str}
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) {{
        return;
    }}
    
    GpuLayoutInfo c_layout = layout_infos[{binding_c}];
    unsigned int temp = idx;
    unsigned int coords[8] = {{0}};
    for (unsigned int d = 0; d < c_layout.ndim; ++d) {{
        coords[d] = temp / c_layout.strides[d];
        temp = temp % c_layout.strides[d];
    }}
    
    {cuda_type} acc = {init_val};
    if (axis_len > 0) {{
        for (unsigned int k = 0; k < axis_len; ++k) {{
            coords[axis] = k;
{offset_calcs}
            {cuda_type} val = {expr_str};
            {update_expr}
        }}
    }}
    
    out[idx] = {final_expr};
}}
"#,
        params_str = params_str,
        binding_c = num_inputs,
        cuda_type = cuda_type,
        init_val = init_val,
        offset_calcs = offset_calcs,
        expr_str = expr_str,
        update_expr = update_expr
    );

    // 6. Get or create kernel module
    let key = format!("fused_reduce_{:?}_{}_{}", op, expr_str, cuda_type);
    let Some(kernel) = get_or_create_kernel(&key, &cuda_src, "fused_reduce_kernel") else {
        return false;
    };

    // 7. Launch
    let mut out_ptr = c.cu_deviceptr();
    let mut layouts_ptr = layout_buf.cu_deviceptr();
    let mut axis_val = axis_value;
    let mut axis_len_val = axis_len_value;
    let mut out_n = out_numel_value;

    let mut in_ptrs: Vec<CUdeviceptr> = inputs
        .iter()
        .map(|input| input.storage().cu_deviceptr())
        .collect();

    let mut args = vec![
        &mut out_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut layouts_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut axis_val as *mut u32 as *mut std::ffi::c_void,
        &mut axis_len_val as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];
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
