#![allow(clippy::too_many_arguments)]

use crate::driver::CudaDriver;
use crate::kernels::{create_layout_buffer, get_cuda_function};
use crate::storage::CudaStorage;
use coeus_core::Layout;

use super::super::validation::{
    checked_numel, cuda_u32, launch_grid_size, layouts_fit_cuda, CUDA_BLOCK_SIZE,
};
/// Launch the 3-D convolution backward kernel on the GPU.
///
/// Computes gradients for input and weight from the 3-D convolution backward pass.
/// Returns `true` if the kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_conv3d_backward(
    grad_out: &CudaStorage<f32>,
    grad_out_layout: &Layout,
    input: &CudaStorage<f32>,
    input_layout: &Layout,
    weight: &CudaStorage<f32>,
    weight_layout: &Layout,
    grad_input: Option<&mut CudaStorage<f32>>,
    grad_input_layout: &Layout,
    grad_weight: Option<&mut CudaStorage<f32>>,
    grad_weight_layout: &Layout,
    grad_bias: Option<&mut CudaStorage<f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> bool {
    if !layouts_fit_cuda(&[
        grad_out_layout,
        input_layout,
        weight_layout,
        grad_input_layout,
        grad_weight_layout,
    ]) {
        return false;
    }
    let Some(mut stride_val) = cuda_u32(stride) else {
        return false;
    };
    let Some(mut pad_val) = cuda_u32(padding) else {
        return false;
    };
    let Some(mut dil_val) = cuda_u32(dilation) else {
        return false;
    };
    let drv = match CudaDriver::get() {
        Some(d) => d,
        None => return false,
    };

    let go_layout_buf = create_layout_buffer(grad_out_layout);
    let mut go_ptr = grad_out.cu_deviceptr();
    let mut gol_ptr = go_layout_buf.cu_deviceptr();

    if let Some(gi) = grad_input {
        let Some(func) = get_cuda_function("conv3d_grad_input_f32") else {
            return false;
        };
        let w_layout_buf = create_layout_buffer(weight_layout);
        let gi_layout_buf = create_layout_buffer(grad_input_layout);

        let mut w_ptr = weight.cu_deviceptr();
        let mut gi_ptr = gi.cu_deviceptr();
        let mut wl_ptr = w_layout_buf.cu_deviceptr();
        let mut gil_ptr = gi_layout_buf.cu_deviceptr();
        let Some(numel_in) = checked_numel(grad_input_layout) else {
            return false;
        };
        let Some(mut num_in) = cuda_u32(numel_in) else {
            return false;
        };

        let mut args: [*mut std::ffi::c_void; 10] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gi_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gol_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut wl_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gil_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut stride_val as *mut u32 as *mut std::ffi::c_void,
            &mut pad_val as *mut u32 as *mut std::ffi::c_void,
            &mut dil_val as *mut u32 as *mut std::ffi::c_void,
            &mut num_in as *mut u32 as *mut std::ffi::c_void,
        ];

        let Some(grid_size) = launch_grid_size(numel_in) else {
            return false;
        };
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gw) = grad_weight {
        let Some(func) = get_cuda_function("conv3d_grad_weight_f32") else {
            return false;
        };
        let in_layout_buf = create_layout_buffer(input_layout);
        let gw_layout_buf = create_layout_buffer(grad_weight_layout);

        let mut in_ptr = input.cu_deviceptr();
        let mut gw_ptr = gw.cu_deviceptr();
        let mut inl_ptr = in_layout_buf.cu_deviceptr();
        let mut gwl_ptr = gw_layout_buf.cu_deviceptr();
        let Some(numel_w) = checked_numel(grad_weight_layout) else {
            return false;
        };
        let Some(mut num_w) = cuda_u32(numel_w) else {
            return false;
        };

        let mut args: [*mut std::ffi::c_void; 10] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gw_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gol_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut inl_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gwl_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut stride_val as *mut u32 as *mut std::ffi::c_void,
            &mut pad_val as *mut u32 as *mut std::ffi::c_void,
            &mut dil_val as *mut u32 as *mut std::ffi::c_void,
            &mut num_w as *mut u32 as *mut std::ffi::c_void,
        ];

        let Some(grid_size) = launch_grid_size(numel_w) else {
            return false;
        };
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gb) = grad_bias {
        let Some(func) = get_cuda_function("conv3d_grad_bias_f32") else {
            return false;
        };
        let mut gb_ptr = gb.cu_deviceptr();
        let Some(c_out) = weight_layout.shape().first().copied() else {
            return false;
        };
        let Some(mut c_out_val) = cuda_u32(c_out) else {
            return false;
        };

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gb_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gol_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut c_out_val as *mut u32 as *mut std::ffi::c_void,
        ];

        let Some(grid_size) = launch_grid_size(c_out) else {
            return false;
        };
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    true
}
