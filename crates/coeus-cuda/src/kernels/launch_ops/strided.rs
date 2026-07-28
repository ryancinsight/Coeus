#![allow(clippy::too_many_arguments)]

use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::{CUdeviceptr, CudaDriver, get_cuda_context};
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout};

/// Launch a strided binary element-wise kernel on the GPU.
///
/// Computes `c = op(a, b)` over `n` elements using the provided layouts for
/// strided indexing. Returns `true` if the kernel launched successfully, `false`
/// if the driver or context is unavailable.
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
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    if !crate::kernels::validation::layouts_fit_cuda(&[a_layout, b_layout, c_layout])
        || !crate::kernels::validation::layout_supports_cuda_output_indexing(c_layout)
        || a_layout.ndim() > c_layout.ndim()
        || b_layout.ndim() > c_layout.ndim()
    {
        return false;
    }
    let Some(n_value) = crate::kernels::validation::cuda_u32(n) else {
        return false;
    };
    let Some(grid_size) = crate::kernels::validation::launch_grid_size(n) else {
        return false;
    };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::BinaryOp::Add => "val_a + val_b",
        coeus_ops::BinaryOp::Sub => "val_a - val_b",
        coeus_ops::BinaryOp::Mul => "val_a * val_b",
        coeus_ops::BinaryOp::Div => "val_a / val_b",
        _ => return false,
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
    const {cuda_type}* b,
    {cuda_type}* c,
    const GpuLayoutInfo* layout_infos,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    GpuLayoutInfo a_layout = layout_infos[0];
    GpuLayoutInfo b_layout = layout_infos[1];
    GpuLayoutInfo c_layout = layout_infos[2];

    unsigned int temp = idx;
    unsigned int off_a = a_layout.offset;
    unsigned int off_b = b_layout.offset;
    unsigned int off_c = c_layout.offset;

    for (unsigned int d = 0; d < c_layout.ndim; ++d) {{
        unsigned int coord = temp / c_layout.strides[d];
        temp = temp % c_layout.strides[d];

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
    let Some(kernel) =
        crate::kernels::fuse::get_or_create_kernel(&key, &cuda_src, "binary_strided_kernel")
    else {
        return false;
    };

    let Ok(a_layout_gpu) = GpuLayoutInfo::try_from(a_layout) else {
        return false;
    };
    let Ok(b_layout_gpu) = GpuLayoutInfo::try_from(b_layout) else {
        return false;
    };
    let Ok(c_layout_gpu) = GpuLayoutInfo::try_from(c_layout) else {
        return false;
    };
    let layouts = [a_layout_gpu, b_layout_gpu, c_layout_gpu];
    let layout_slice: &[u32] = bytemuck::cast_slice(&layouts);
    let mut layout_buf = CudaStorage::<u32>::new(layout_slice.len());
    CudaBackend::new().copy_to_device(layout_slice, &mut layout_buf);

    let mut a_ptr = a.cu_deviceptr();
    let mut b_ptr = b.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut layouts_ptr = layout_buf.cu_deviceptr();
    let mut n_val = n_value;

    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut b_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut layouts_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
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

/// Launch a strided unary element-wise kernel on the GPU.
///
/// Computes `c = op(a)` over `n` elements using the provided layouts for strided
/// indexing. Returns `true` if the kernel launched successfully, `false` if the
/// driver or context is unavailable.
pub fn launch_strided_unary<T: CudaScalar>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
    n: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    if !crate::kernels::validation::layouts_fit_cuda(&[a_layout, c_layout])
        || !crate::kernels::validation::layout_supports_cuda_output_indexing(c_layout)
        || a_layout.ndim() > c_layout.ndim()
    {
        return false;
    }
    let Some(n_value) = crate::kernels::validation::cuda_u32(n) else {
        return false;
    };
    let Some(grid_size) = crate::kernels::validation::launch_grid_size(n) else {
        return false;
    };
    let cuda_type = T::CUDA_TYPE;

    let op_expr = match op {
        coeus_ops::UnaryOp::Relu => "(val_a > 0.0f) ? val_a : 0.0f",
        coeus_ops::UnaryOp::ReluGrad => "(val_a > 0.0f) ? 1.0f : 0.0f",
        coeus_ops::UnaryOp::Sigmoid => "1.0f / (1.0f + expf(-val_a))",
        coeus_ops::UnaryOp::SigmoidGrad => "val_a * (1.0f - val_a)",
        coeus_ops::UnaryOp::Tanh => "tanhf(val_a)",
        coeus_ops::UnaryOp::TanhGrad => "1.0f - val_a * val_a",
        // Exact erf GELU, matching the CPU `gelu_op` and WGPU contract
        // (0.5 x (1 + erf(x/sqrt(2)))). 0.70710678f = 1/sqrt(2).
        coeus_ops::UnaryOp::Gelu => "0.5f * val_a * (1.0f + erff(val_a * 0.70710678f))",
        // d/dx of exact GELU: 0.5(1 + erf(x/sqrt(2))) + x/sqrt(2pi) exp(-x^2/2).
        // 0.3989422804f = 1/sqrt(2pi).
        coeus_ops::UnaryOp::GeluGrad => {
            "0.5f * (1.0f + erff(val_a * 0.70710678f)) + val_a * 0.3989422804f * expf(-0.5f * val_a * val_a)"
        }
        coeus_ops::UnaryOp::Sin => "sinf(val_a)",
        coeus_ops::UnaryOp::Cos => "cosf(val_a)",
        coeus_ops::UnaryOp::Exp => "expf(val_a)",
        coeus_ops::UnaryOp::Log => "logf(val_a)",
        coeus_ops::UnaryOp::Erf => "erff(val_a)",
        coeus_ops::UnaryOp::Erfc => "erfcf(val_a)",
        coeus_ops::UnaryOp::Tan => "tanf(val_a)",
        coeus_ops::UnaryOp::Asin => "asinf(val_a)",
        coeus_ops::UnaryOp::Acos => "acosf(val_a)",
        coeus_ops::UnaryOp::Atan => "atanf(val_a)",
        coeus_ops::UnaryOp::Sinh => "sinhf(val_a)",
        coeus_ops::UnaryOp::Cosh => "coshf(val_a)",
        coeus_ops::UnaryOp::Log2 => "log2f(val_a)",
        coeus_ops::UnaryOp::Log10 => "log10f(val_a)",
        coeus_ops::UnaryOp::Exp2 => "exp2f(val_a)",
        coeus_ops::UnaryOp::Atanh => "atanhf(val_a)",
        coeus_ops::UnaryOp::Asinh => "asinhf(val_a)",
        coeus_ops::UnaryOp::Acosh => "acoshf(val_a)",
        coeus_ops::UnaryOp::Expm1 => "expm1f(val_a)",
        coeus_ops::UnaryOp::Log1p => "log1pf(val_a)",
        coeus_ops::UnaryOp::Neg => "-val_a",
        coeus_ops::UnaryOp::Abs => "fabsf(val_a)",
        coeus_ops::UnaryOp::Sqrt => "sqrtf(val_a)",
        coeus_ops::UnaryOp::Silu => "val_a / (1.0f + expf(-val_a))",
        coeus_ops::UnaryOp::SiluGrad => {
            "(1.0f / (1.0f + expf(-val_a))) * (1.0f + val_a * (1.0f - (1.0f / (1.0f + expf(-val_a)))))"
        }
        coeus_ops::UnaryOp::Mish => "val_a * tanhf(logf(1.0f + expf(val_a)))",
        coeus_ops::UnaryOp::MishGrad => {
            "tanhf(logf(1.0f + expf(val_a))) + val_a * (1.0f - tanhf(logf(1.0f + expf(val_a))) * tanhf(logf(1.0f + expf(val_a)))) * (1.0f / (1.0f + expf(-val_a)))"
        }
        coeus_ops::UnaryOp::Recip => "1.0f / val_a",
        coeus_ops::UnaryOp::Sign => "(val_a > 0.0f) ? 1.0f : ((val_a < 0.0f) ? -1.0f : 0.0f)",
        coeus_ops::UnaryOp::Floor => "floorf(val_a)",
        coeus_ops::UnaryOp::Ceil => "ceilf(val_a)",
        // rintf = ties-to-even (matches torch.round / CPU round_ties_even).
        coeus_ops::UnaryOp::Round => "rintf(val_a)",
        coeus_ops::UnaryOp::Trunc => "truncf(val_a)",
        _ => return false,
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
    {cuda_type}* c,
    const GpuLayoutInfo* layout_infos,
    unsigned int n
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    GpuLayoutInfo a_layout = layout_infos[0];
    GpuLayoutInfo c_layout = layout_infos[1];

    unsigned int temp = idx;
    unsigned int off_a = a_layout.offset;
    unsigned int off_c = c_layout.offset;

    for (unsigned int d = 0; d < c_layout.ndim; ++d) {{
        unsigned int coord = temp / c_layout.strides[d];
        temp = temp % c_layout.strides[d];

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
    let Some(kernel) =
        crate::kernels::fuse::get_or_create_kernel(&key, &cuda_src, "unary_strided_kernel")
    else {
        return false;
    };

    let Ok(a_layout_gpu) = GpuLayoutInfo::try_from(a_layout) else {
        return false;
    };
    let Ok(c_layout_gpu) = GpuLayoutInfo::try_from(c_layout) else {
        return false;
    };
    let layouts = [a_layout_gpu, c_layout_gpu];
    let layout_slice: &[u32] = bytemuck::cast_slice(&layouts);
    let mut layout_buf = CudaStorage::<u32>::new(layout_slice.len());
    CudaBackend::new().copy_to_device(layout_slice, &mut layout_buf);

    let mut a_ptr = a.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();
    let mut layouts_ptr = layout_buf.cu_deviceptr();
    let mut n_val = n_value;

    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut a_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut c_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut layouts_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
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
