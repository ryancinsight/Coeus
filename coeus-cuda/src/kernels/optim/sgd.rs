use crate::driver::{get_cuda_context, CudaDriver};
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::Layout;

pub fn launch_sgd_step(
    param: &mut CudaStorage<f32>,
    param_layout: &Layout,
    grad: &CudaStorage<f32>,
    grad_layout: &Layout,
    velocity: &mut CudaStorage<f32>,
    velocity_layout: &Layout,
    lr: f32,
    momentum: f32,
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
        && velocity_layout.is_contiguous();

    let mut param_ptr = param.cu_deviceptr();
    let mut grad_ptr = grad.cu_deviceptr();
    let mut velocity_ptr = velocity.cu_deviceptr();

    if is_contiguous {
        let cuda_src = r#"
extern "C" __global__ void sgd_contiguous_kernel(
    float* param,
    const float* grad,
    float* velocity,
    float lr,
    float momentum,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float v = velocity[idx] * momentum + g;
    velocity[idx] = v;
    param[idx] -= lr * v;
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "sgd_contiguous",
            cuda_src,
            "sgd_contiguous_kernel",
        ) else {
            return false;
        };

        let mut lr_val = lr;
        let mut momentum_val = momentum;
        let mut n_val = n as u32;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut velocity_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut momentum_val as *mut f32 as *mut std::ffi::c_void,
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

extern "C" __global__ void sgd_strided_kernel(
    float* param,
    GpuLayoutInfo param_layout,
    const float* grad,
    GpuLayoutInfo grad_layout,
    float* velocity,
    GpuLayoutInfo velocity_layout,
    float lr,
    float momentum,
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
    unsigned int off_v = velocity_layout.offset;

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
        if (d >= param_layout.ndim - velocity_layout.ndim) {
            unsigned int vd = d + velocity_layout.ndim - param_layout.ndim;
            if (velocity_layout.shape[vd] > 1) {
                off_v += coord * velocity_layout.strides[vd];
            }
        }
    }

    float g = grad[off_g];
    float v = velocity[off_v] * momentum + g;
    velocity[off_v] = v;
    param[off_p] -= lr * v;
}
"#;
        let Some(kernel) = crate::kernels::fuse::get_or_create_kernel(
            "sgd_strided",
            cuda_src,
            "sgd_strided_kernel",
        ) else {
            return false;
        };

        let gpu_param_layout = GpuLayoutInfo::from_layout(param_layout);
        let gpu_grad_layout = GpuLayoutInfo::from_layout(grad_layout);
        let gpu_velocity_layout = GpuLayoutInfo::from_layout(velocity_layout);

        let mut lr_val = lr;
        let mut momentum_val = momentum;
        let mut n_val = n as u32;

        let mut args: [*mut std::ffi::c_void; 9] = [
            &mut param_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_param_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut grad_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_grad_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut velocity_ptr as *mut u64 as *mut std::ffi::c_void,
            &gpu_velocity_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
            &mut lr_val as *mut f32 as *mut std::ffi::c_void,
            &mut momentum_val as *mut f32 as *mut std::ffi::c_void,
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
