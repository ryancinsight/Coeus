#![allow(clippy::too_many_arguments)]

use crate::backend::CudaScalar;
use crate::driver::{get_cuda_context, CUdeviceptr, CudaDriver};
use crate::storage::CudaStorage;

/// Launch a contiguous binary element-wise kernel on the GPU.
///
/// Computes `c = op(a, b)` over `n` contiguous elements. Returns `true` if the
/// kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_contiguous_binary<T: CudaScalar>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    let Some(n_value) = crate::kernels::validation::cuda_u32(n) else {
        return false;
    };
    let Some(grid_size) = crate::kernels::validation::launch_grid_size(n) else {
        return false;
    };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::BinaryOp::Add => "a[idx] + b[idx]",
        coeus_ops::BinaryOp::Sub => "a[idx] - b[idx]",
        coeus_ops::BinaryOp::Mul => "a[idx] * b[idx]",
        coeus_ops::BinaryOp::Div => "a[idx] / b[idx]",
        _ => return false,
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
    let Some(kernel) =
        crate::kernels::fuse::get_or_create_kernel(&key, &cuda_src, "contiguous_binary_kernel")
    else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut b_ptr = b.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n_value;

    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut b_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size,
            1,
            1,
            crate::kernels::validation::CUDA_BLOCK_SIZE,
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

/// Launch a contiguous unary element-wise kernel on the GPU.
///
/// Computes `c = op(a)` over `n` contiguous elements. Returns `true` if the
/// kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_contiguous_unary<T: CudaScalar>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    let Some(n_value) = crate::kernels::validation::cuda_u32(n) else {
        return false;
    };
    let Some(grid_size) = crate::kernels::validation::launch_grid_size(n) else {
        return false;
    };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::UnaryOp::Relu => "(a[idx] > 0.0f) ? a[idx] : 0.0f",
        coeus_ops::UnaryOp::ReluGrad => "(a[idx] > 0.0f) ? 1.0f : 0.0f",
        coeus_ops::UnaryOp::Sigmoid => "1.0f / (1.0f + expf(-a[idx]))",
        coeus_ops::UnaryOp::SigmoidGrad => "a[idx] * (1.0f - a[idx])",
        coeus_ops::UnaryOp::Tanh => "tanhf(a[idx])",
        coeus_ops::UnaryOp::TanhGrad => "1.0f - a[idx] * a[idx]",
        // Exact erf GELU, matching the CPU `gelu_op` and WGPU contract
        // (0.5 x (1 + erf(x/sqrt(2)))). 0.70710678f = 1/sqrt(2).
        coeus_ops::UnaryOp::Gelu => "0.5f * a[idx] * (1.0f + erff(a[idx] * 0.70710678f))",
        // d/dx of exact GELU: 0.5(1 + erf(x/sqrt(2))) + x/sqrt(2pi) exp(-x^2/2).
        // 0.3989422804f = 1/sqrt(2pi).
        coeus_ops::UnaryOp::GeluGrad => "0.5f * (1.0f + erff(a[idx] * 0.70710678f)) + a[idx] * 0.3989422804f * expf(-0.5f * a[idx] * a[idx])",
        coeus_ops::UnaryOp::Sin => "sinf(a[idx])",
        coeus_ops::UnaryOp::Cos => "cosf(a[idx])",
        coeus_ops::UnaryOp::Exp => "expf(a[idx])",
        coeus_ops::UnaryOp::Log => "logf(a[idx])",
        coeus_ops::UnaryOp::Erf => "erff(a[idx])",
        coeus_ops::UnaryOp::Erfc => "erfcf(a[idx])",
        coeus_ops::UnaryOp::Lgamma => "lgammaf(a[idx])",
        coeus_ops::UnaryOp::Tan => "tanf(a[idx])",
        coeus_ops::UnaryOp::Asin => "asinf(a[idx])",
        coeus_ops::UnaryOp::Acos => "acosf(a[idx])",
        coeus_ops::UnaryOp::Atan => "atanf(a[idx])",
        coeus_ops::UnaryOp::Sinh => "sinhf(a[idx])",
        coeus_ops::UnaryOp::Cosh => "coshf(a[idx])",
        coeus_ops::UnaryOp::Log2 => "log2f(a[idx])",
        coeus_ops::UnaryOp::Log10 => "log10f(a[idx])",
        coeus_ops::UnaryOp::Exp2 => "exp2f(a[idx])",
        coeus_ops::UnaryOp::Atanh => "atanhf(a[idx])",
        coeus_ops::UnaryOp::Asinh => "asinhf(a[idx])",
        coeus_ops::UnaryOp::Acosh => "acoshf(a[idx])",
        coeus_ops::UnaryOp::Expm1 => "expm1f(a[idx])",
        coeus_ops::UnaryOp::Log1p => "log1pf(a[idx])",
        coeus_ops::UnaryOp::Neg => "-a[idx]",
        coeus_ops::UnaryOp::Abs => "fabsf(a[idx])",
        coeus_ops::UnaryOp::Sqrt => "sqrtf(a[idx])",
        coeus_ops::UnaryOp::Silu => "a[idx] / (1.0f + expf(-a[idx]))",
        coeus_ops::UnaryOp::SiluGrad => "(1.0f / (1.0f + expf(-a[idx]))) * (1.0f + a[idx] * (1.0f - (1.0f / (1.0f + expf(-a[idx])))))",
        coeus_ops::UnaryOp::Mish => "a[idx] * tanhf(logf(1.0f + expf(a[idx])))",
        coeus_ops::UnaryOp::MishGrad => "tanhf(logf(1.0f + expf(a[idx]))) + a[idx] * (1.0f - tanhf(logf(1.0f + expf(a[idx]))) * tanhf(logf(1.0f + expf(a[idx])))) * (1.0f / (1.0f + expf(-a[idx])))",
        coeus_ops::UnaryOp::Recip => "1.0f / a[idx]",
        coeus_ops::UnaryOp::Sign => "(a[idx] > 0.0f) ? 1.0f : ((a[idx] < 0.0f) ? -1.0f : 0.0f)",
        coeus_ops::UnaryOp::Floor => "floorf(a[idx])",
        coeus_ops::UnaryOp::Ceil => "ceilf(a[idx])",
        // rintf = ties-to-even (matches torch.round / CPU round_ties_even).
        coeus_ops::UnaryOp::Round => "rintf(a[idx])",
        coeus_ops::UnaryOp::Trunc => "truncf(a[idx])",
        _ => return false,
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
    let Some(kernel) =
        crate::kernels::fuse::get_or_create_kernel(&key, &cuda_src, "contiguous_unary_kernel")
    else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut n_val = n_value;

    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size,
            1,
            1,
            crate::kernels::validation::CUDA_BLOCK_SIZE,
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
