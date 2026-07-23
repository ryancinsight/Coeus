use crate::driver::{get_cuda_context, CudaDriver};
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::Layout;

/// Launch the AdaGrad optimizer step kernel on the GPU.
///
/// Updates parameters using the AdaGrad accumulation rule. Returns `true` if the
/// kernel launched successfully, `false` if the driver or context is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn launch_adagrad_step(
    param: &mut CudaStorage<f32>,
    param_layout: &Layout,
    grad: &CudaStorage<f32>,
    grad_layout: &Layout,
    history: &mut CudaStorage<f32>,
    history_layout: &Layout,
    lr: f32,
    eps: f32,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };

    let n = param_layout.numel();
    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && history_layout.is_contiguous();

    let mut param_ptr = param.cu_deviceptr();
    let mut grad_ptr = grad.cu_deviceptr();
    let mut history_ptr = history.cu_deviceptr();

    if is_contiguous {
        let cuda_src = r#"
extern "C" __global__ void adagrad_contiguous_kernel(
    float* param,
    const float* grad,
    float* history,
    float lr,
    float eps,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float h = history[idx] + g * g;
    history[idx] = h;
    param[idx] -= lr * g / (sqrtf(h) + eps);
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "adagrad_contiguous",
            cuda_src,
            "adagrad_contiguous_kernel",
        ) else {
            return false;
        };

        let mut lr_val = lr;
        let mut eps_val = eps;
        let mut n_val = n as u32;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut history_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut eps_val as *mut f32 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
        ];

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

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
    } else {
        let cuda_src = r#"
struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

extern "C" __global__ void adagrad_strided_kernel(
    float* param,
    GpuLayoutInfo param_layout,
    const float* grad,
    GpuLayoutInfo grad_layout,
    float* history,
    GpuLayoutInfo history_layout,
    float lr,
    float eps,
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
    unsigned int off_h = history_layout.offset;

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
        if (d >= param_layout.ndim - history_layout.ndim) {
            unsigned int hd = d + history_layout.ndim - param_layout.ndim;
            if (history_layout.shape[hd] > 1) {
                off_h += coord * history_layout.strides[hd];
            }
        }
    }

    float g = grad[off_g];
    float h = history[off_h] + g * g;
    history[off_h] = h;
    param[off_p] -= lr * g / (sqrtf(h) + eps);
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "adagrad_strided",
            cuda_src,
            "adagrad_strided_kernel",
        ) else {
            return false;
        };

        let Ok(gpu_param_layout) = GpuLayoutInfo::try_from(param_layout) else {
            return false;
        };
        let Ok(gpu_grad_layout) = GpuLayoutInfo::try_from(grad_layout) else {
            return false;
        };
        let Ok(gpu_history_layout) = GpuLayoutInfo::try_from(history_layout) else {
            return false;
        };

        let mut lr_val = lr;
        let mut eps_val = eps;
        let mut n_val = n as u32;

        let mut args: [*mut std::ffi::c_void; 9] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_param_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_grad_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut history_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_history_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut eps_val as *mut f32 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
        ];

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

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
}
