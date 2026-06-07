#![allow(clippy::too_many_arguments)]

use coeus_core::Layout;
use crate::storage::CudaStorage;
use crate::driver::{CudaDriver, CUdeviceptr, get_cuda_context};
use crate::backend::CudaScalar;
use super::GpuLayoutInfo;

pub fn launch_contiguous_binary<T: CudaScalar>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else { return false; };
    let Some(_ctx) = get_cuda_context() else { return false; };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::BinaryOp::Add => "a[idx] + b[idx]",
        coeus_ops::BinaryOp::Sub => "a[idx] - b[idx]",
        coeus_ops::BinaryOp::Mul => "a[idx] * b[idx]",
        coeus_ops::BinaryOp::Div => "a[idx] / b[idx]",
    };

    let cuda_src = format!(
        r#"
extern "C" __global__ void contiguous_binary_kernel(
    const {cuda_type}* a,
    const {cuda_type}* b,
    {cuda_type}* c,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = {op_expr};
    }}
}}
"#,
        cuda_type = cuda_type,
        op_expr = op_expr
    );

    let key = format!("contiguous_binary_{:?}_{}", op, cuda_type);
    let Some(kernel) = super::fuse::get_or_create_kernel(&key, &cuda_src, "contiguous_binary_kernel") else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut b_ptr = b.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n as u32;

    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut b_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32, 1, 1,
            block_size as u32, 1, 1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}

pub fn launch_contiguous_unary<T: CudaScalar>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else { return false; };
    let Some(_ctx) = get_cuda_context() else { return false; };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::UnaryOp::Relu => "(a[idx] > 0.0f) ? a[idx] : 0.0f",
        coeus_ops::UnaryOp::ReluGrad => "(a[idx] > 0.0f) ? 1.0f : 0.0f",
        coeus_ops::UnaryOp::Sigmoid => "1.0f / (1.0f + expf(-a[idx]))",
        coeus_ops::UnaryOp::SigmoidGrad => "a[idx] * (1.0f - a[idx])",
        coeus_ops::UnaryOp::Tanh => "tanhf(a[idx])",
        coeus_ops::UnaryOp::TanhGrad => "1.0f - a[idx] * a[idx]",
        coeus_ops::UnaryOp::Gelu => "0.5f * a[idx] * (1.0f + tanhf(0.79788456f * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx])))",
        coeus_ops::UnaryOp::GeluGrad => "0.5f * (1.0f + tanhf(0.79788456f * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx]))) + 0.5f * a[idx] * (1.0f - tanhf(0.79788456f * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx])) * tanhf(0.79788456f * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx]))) * 0.79788456f * (1.0f + 0.134145f * a[idx] * a[idx])",
        coeus_ops::UnaryOp::Sin => "sinf(a[idx])",
        coeus_ops::UnaryOp::Cos => "cosf(a[idx])",
        coeus_ops::UnaryOp::Exp => "expf(a[idx])",
        coeus_ops::UnaryOp::Log => "logf(a[idx])",
        coeus_ops::UnaryOp::Neg => "-a[idx]",
        coeus_ops::UnaryOp::Abs => "fabsf(a[idx])",
        coeus_ops::UnaryOp::Sqrt => "sqrtf(a[idx])",
        coeus_ops::UnaryOp::Silu => "a[idx] / (1.0f + expf(-a[idx]))",
        coeus_ops::UnaryOp::SiluGrad => "(1.0f / (1.0f + expf(-a[idx]))) * (1.0f + a[idx] * (1.0f - (1.0f / (1.0f + expf(-a[idx])))))",
        coeus_ops::UnaryOp::Mish => "a[idx] * tanhf(logf(1.0f + expf(a[idx])))",
        coeus_ops::UnaryOp::MishGrad => "tanhf(logf(1.0f + expf(a[idx]))) + a[idx] * (1.0f - tanhf(logf(1.0f + expf(a[idx]))) * tanhf(logf(1.0f + expf(a[idx])))) * (1.0f / (1.0f + expf(-a[idx])))",
    };

    let cuda_src = format!(
        r#"
extern "C" __global__ void contiguous_unary_kernel(
    const {cuda_type}* a,
    {cuda_type}* c,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = {op_expr};
    }}
}}
"#,
        cuda_type = cuda_type,
        op_expr = op_expr
    );

    let key = format!("contiguous_unary_{:?}_{}", op, cuda_type);
    let Some(kernel) = super::fuse::get_or_create_kernel(&key, &cuda_src, "contiguous_unary_kernel") else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n as u32;

    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32, 1, 1,
            block_size as u32, 1, 1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}

pub fn launch_strided_binary<T: CudaScalar>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    b: &CudaStorage<T>,
    b_layout: &Layout,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else { return false; };
    let Some(_ctx) = get_cuda_context() else { return false; };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::BinaryOp::Add => "val_a + val_b",
        coeus_ops::BinaryOp::Sub => "val_a - val_b",
        coeus_ops::BinaryOp::Mul => "val_a * val_b",
        coeus_ops::BinaryOp::Div => "val_a / val_b",
    };

    let cuda_src = format!(
        r#"
struct GpuLayoutInfo {{
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
}};

extern "C" __global__ void binary_strided_kernel(
    const {cuda_type}* a,
    GpuLayoutInfo a_layout,
    const {cuda_type}* b,
    GpuLayoutInfo b_layout,
    {cuda_type}* c,
    GpuLayoutInfo c_layout,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int c_contig_strides[8];
    unsigned int accum = 1;
    for (int d = (int)c_layout.ndim - 1; d >= 0; --d) {{
        c_contig_strides[d] = accum;
        accum *= c_layout.shape[d];
    }}
    
    unsigned int temp = idx;
    unsigned int off_a = a_layout.offset;
    unsigned int off_b = b_layout.offset;
    unsigned int off_c = c_layout.offset;
    
    for (unsigned int d = 0; d < c_layout.ndim; ++d) {{
        unsigned int coord = temp / c_contig_strides[d];
        temp = temp % c_contig_strides[d];
        
        off_c += coord * c_layout.strides[d];
        
        if (d >= c_layout.ndim - a_layout.ndim) {{
            unsigned int ad = d + a_layout.ndim - c_layout.ndim;
            if (a_layout.shape[ad] > 1) {{
                off_a += coord * a_layout.strides[ad];
            }}
        }}
        if (d >= c_layout.ndim - b_layout.ndim) {{
            unsigned int bd = d + b_layout.ndim - c_layout.ndim;
            if (b_layout.shape[bd] > 1) {{
                off_b += coord * b_layout.strides[bd];
            }}
        }}
    }}
    
    {cuda_type} val_a = a[off_a];
    {cuda_type} val_b = b[off_b];
    c[off_c] = {op_expr};
}}
"#,
        cuda_type = cuda_type,
        op_expr = op_expr
    );

    let key = format!("strided_binary_{:?}_{}", op, cuda_type);
    let Some(kernel) = super::fuse::get_or_create_kernel(&key, &cuda_src, "binary_strided_kernel") else {
        return false;
    };

    let gpu_a_layout = GpuLayoutInfo::from_layout(a_layout);
    let gpu_b_layout = GpuLayoutInfo::from_layout(b_layout);
    let gpu_c_layout = GpuLayoutInfo::from_layout(c_layout);

    let mut a_ptr = a.cu_deviceptr();
    let mut b_ptr = b.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n as u32;

    let mut args: [*mut std::ffi::c_void; 7] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_a_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut b_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_b_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_c_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32, 1, 1,
            block_size as u32, 1, 1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}

pub fn launch_strided_unary<T: CudaScalar>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else { return false; };
    let Some(_ctx) = get_cuda_context() else { return false; };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::UnaryOp::Relu => "(val_a > 0.0f) ? val_a : 0.0f",
        coeus_ops::UnaryOp::ReluGrad => "(val_a > 0.0f) ? 1.0f : 0.0f",
        coeus_ops::UnaryOp::Sigmoid => "1.0f / (1.0f + expf(-val_a))",
        coeus_ops::UnaryOp::SigmoidGrad => "val_a * (1.0f - val_a)",
        coeus_ops::UnaryOp::Tanh => "tanhf(val_a)",
        coeus_ops::UnaryOp::TanhGrad => "1.0f - val_a * val_a",
        coeus_ops::UnaryOp::Gelu => "0.5f * val_a * (1.0f + tanhf(0.79788456f * (val_a + 0.044715f * val_a * val_a * val_a)))",
        coeus_ops::UnaryOp::GeluGrad => "0.5f * (1.0f + tanhf(0.79788456f * (val_a + 0.044715f * val_a * val_a * val_a))) + 0.5f * val_a * (1.0f - tanhf(0.79788456f * (val_a + 0.044715f * val_a * val_a * val_a)) * tanhf(0.79788456f * (val_a + 0.044715f * val_a * val_a * val_a))) * 0.79788456f * (1.0f + 0.134145f * val_a * val_a)",
        coeus_ops::UnaryOp::Sin => "sinf(val_a)",
        coeus_ops::UnaryOp::Cos => "cosf(val_a)",
        coeus_ops::UnaryOp::Exp => "expf(val_a)",
        coeus_ops::UnaryOp::Log => "logf(val_a)",
        coeus_ops::UnaryOp::Neg => "-val_a",
        coeus_ops::UnaryOp::Abs => "fabsf(val_a)",
        coeus_ops::UnaryOp::Sqrt => "sqrtf(val_a)",
        coeus_ops::UnaryOp::Silu => "val_a / (1.0f + expf(-val_a))",
        coeus_ops::UnaryOp::SiluGrad => "(1.0f / (1.0f + expf(-val_a))) * (1.0f + val_a * (1.0f - (1.0f / (1.0f + expf(-val_a)))))",
        coeus_ops::UnaryOp::Mish => "val_a * tanhf(logf(1.0f + expf(val_a)))",
        coeus_ops::UnaryOp::MishGrad => "tanhf(logf(1.0f + expf(val_a))) + val_a * (1.0f - tanhf(logf(1.0f + expf(val_a))) * tanhf(logf(1.0f + expf(val_a)))) * (1.0f / (1.0f + expf(-val_a)))",
    };

    let cuda_src = format!(
        r#"
struct GpuLayoutInfo {{
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
}};

extern "C" __global__ void unary_strided_kernel(
    const {cuda_type}* a,
    GpuLayoutInfo a_layout,
    {cuda_type}* c,
    GpuLayoutInfo c_layout,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int c_contig_strides[8];
    unsigned int accum = 1;
    for (int d = (int)c_layout.ndim - 1; d >= 0; --d) {{
        c_contig_strides[d] = accum;
        accum *= c_layout.shape[d];
    }}
    
    unsigned int temp = idx;
    unsigned int off_a = a_layout.offset;
    unsigned int off_c = c_layout.offset;
    
    for (unsigned int d = 0; d < c_layout.ndim; ++d) {{
        unsigned int coord = temp / c_contig_strides[d];
        temp = temp % c_contig_strides[d];
        
        off_c += coord * c_layout.strides[d];
        
        if (d >= c_layout.ndim - a_layout.ndim) {{
            unsigned int ad = d + a_layout.ndim - c_layout.ndim;
            if (a_layout.shape[ad] > 1) {{
                off_a += coord * a_layout.strides[ad];
            }}
        }}
    }}
    
    {cuda_type} val_a = a[off_a];
    c[off_c] = {op_expr};
}}
"#,
        cuda_type = cuda_type,
        op_expr = op_expr
    );

    let key = format!("strided_unary_{:?}_{}", op, cuda_type);
    let Some(kernel) = super::fuse::get_or_create_kernel(&key, &cuda_src, "unary_strided_kernel") else {
        return false;
    };

    let gpu_a_layout = GpuLayoutInfo::from_layout(a_layout);
    let gpu_c_layout = GpuLayoutInfo::from_layout(c_layout);

    let mut a_ptr = a.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n as u32;

    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_a_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_c_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32, 1, 1,
            block_size as u32, 1, 1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}
