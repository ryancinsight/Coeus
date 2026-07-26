use crate::driver::{get_cuda_context, CudaDriver};
use crate::kernels::validation::{
    checked_numel, cuda_u32, launch_grid_size, layouts_fit_cuda, layouts_share_shape,
    CUDA_BLOCK_SIZE,
};
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::Layout;

/// Launch the Adam optimizer step kernel on the GPU.
///
/// Updates parameters using the Adam first/second moment estimates. Returns `true` if the
/// kernel launched successfully, `false` if the driver or context is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn launch_adam_step(
    param: &mut CudaStorage<f32>,
    param_layout: &Layout,
    grad: &CudaStorage<f32>,
    grad_layout: &Layout,
    m: &mut CudaStorage<f32>,
    m_layout: &Layout,
    v: &mut CudaStorage<f32>,
    v_layout: &Layout,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };

    let Some(n) = checked_numel(param_layout) else {
        return false;
    };
    let Some(n_value) = cuda_u32(n) else {
        return false;
    };
    let Some(grid_size) = launch_grid_size(n) else {
        return false;
    };
    let Ok(t_value) = i32::try_from(t) else {
        return false;
    };
    if !layouts_fit_cuda(&[param_layout, grad_layout, m_layout, v_layout])
        || !layouts_share_shape(&[param_layout, grad_layout, m_layout, v_layout])
    {
        return false;
    }
    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && m_layout.is_contiguous()
        && v_layout.is_contiguous();

    let mut param_ptr = param.cu_deviceptr();
    let mut grad_ptr = grad.cu_deviceptr();
    let mut m_ptr = m.cu_deviceptr();
    let mut v_ptr = v.cu_deviceptr();

    let bias_correction1 = 1.0f32 - beta1.powi(t_value);
    let bias_correction2 = 1.0f32 - beta2.powi(t_value);

    if is_contiguous {
        let cuda_src = r#"
extern "C" __global__ void adam_contiguous_kernel(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float bias_correction2,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float m_val = m[idx] * beta1 + (1.0f - beta1) * g;
    float v_val = v[idx] * beta2 + (1.0f - beta2) * g * g;
    m[idx] = m_val;
    v[idx] = v_val;
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;
    float denom = sqrtf(v_hat) + eps;
    param[idx] -= lr * m_hat / denom;
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "adam_contiguous",
            cuda_src,
            "adam_contiguous_kernel",
        ) else {
            return false;
        };

        let mut lr_val = lr;
        let mut beta1_val = beta1;
        let mut beta2_val = beta2;
        let mut eps_val = eps;
        let mut bc1_val = bias_correction1;
        let mut bc2_val = bias_correction2;
        let mut n_val = n_value;

        let mut args: [*mut std::ffi::c_void; 11] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut m_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut beta1_val as *mut f32 as *mut std::ffi::c_void,
            &mut beta2_val as *mut f32 as *mut std::ffi::c_void,
            &mut eps_val as *mut f32 as *mut std::ffi::c_void,
            &mut bc1_val as *mut f32 as *mut std::ffi::c_void,
            &mut bc2_val as *mut f32 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
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
    } else {
        let cuda_src = r#"
struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

extern "C" __global__ void adam_strided_kernel(
    float* param,
    GpuLayoutInfo param_layout,
    const float* grad,
    GpuLayoutInfo grad_layout,
    float* m,
    GpuLayoutInfo m_layout,
    float* v,
    GpuLayoutInfo v_layout,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float bias_correction2,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int p_contig_strides[8];
    unsigned int accum = 1;
    for (int d = (int)param_layout.ndim - 1; d >= 0; --d) {
        p_contig_strides[d] = accum;
        accum *= param_layout.shape[d];
    }

    unsigned int temp = idx;
    unsigned int off_p = param_layout.offset;
    unsigned int off_g = grad_layout.offset;
    unsigned int off_m = m_layout.offset;
    unsigned int off_v = v_layout.offset;

    for (unsigned int d = 0; d < param_layout.ndim; ++d) {
        unsigned int coord = temp / p_contig_strides[d];
        temp = temp % p_contig_strides[d];

        off_p += coord * param_layout.strides[d];

        if (d >= param_layout.ndim - grad_layout.ndim) {
            unsigned int gd = d + grad_layout.ndim - param_layout.ndim;
            if (grad_layout.shape[gd] > 1) {
                off_g += coord * grad_layout.strides[gd];
            }
        }
        if (d >= param_layout.ndim - m_layout.ndim) {
            unsigned int md = d + m_layout.ndim - param_layout.ndim;
            if (m_layout.shape[md] > 1) {
                off_m += coord * m_layout.strides[md];
            }
        }
        if (d >= param_layout.ndim - v_layout.ndim) {
            unsigned int vd = d + v_layout.ndim - param_layout.ndim;
            if (v_layout.shape[vd] > 1) {
                off_v += coord * v_layout.strides[vd];
            }
        }
    }

    float g = grad[off_g];
    float m_val = m[off_m] * beta1 + (1.0f - beta1) * g;
    float v_val = v[off_v] * beta2 + (1.0f - beta2) * g * g;
    m[off_m] = m_val;
    v[off_v] = v_val;
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;
    float denom = sqrtf(v_hat) + eps;
    param[off_p] -= lr * m_hat / denom;
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "adam_strided",
            cuda_src,
            "adam_strided_kernel",
        ) else {
            return false;
        };

        let Ok(gpu_param_layout) = GpuLayoutInfo::try_from(param_layout) else {
            return false;
        };
        let Ok(gpu_grad_layout) = GpuLayoutInfo::try_from(grad_layout) else {
            return false;
        };
        let Ok(gpu_m_layout) = GpuLayoutInfo::try_from(m_layout) else {
            return false;
        };
        let Ok(gpu_v_layout) = GpuLayoutInfo::try_from(v_layout) else {
            return false;
        };

        let mut lr_val = lr;
        let mut beta1_val = beta1;
        let mut beta2_val = beta2;
        let mut eps_val = eps;
        let mut bc1_val = bias_correction1;
        let mut bc2_val = bias_correction2;
        let mut n_val = n_value;

        let mut args: [*mut std::ffi::c_void; 15] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_param_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_grad_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut m_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_m_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_v_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut beta1_val as *mut f32 as *mut std::ffi::c_void,
            &mut beta2_val as *mut f32 as *mut std::ffi::c_void,
            &mut eps_val as *mut f32 as *mut std::ffi::c_void,
            &mut bc1_val as *mut f32 as *mut std::ffi::c_void,
            &mut bc2_val as *mut f32 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
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
}
