#![allow(clippy::too_many_arguments)]

use crate::driver::CudaDriver;
use crate::kernels::{create_layout_buffer, get_cuda_function};
use crate::storage::CudaStorage;
use coeus_core::Layout;

use crate::kernels::validation::{CUDA_BLOCK_SIZE, cuda_u32, launch_grid_size, layouts_fit_cuda};
/// Launch the 1-D convolution kernel on the GPU.
///
/// Computes forward 1-D convolution with optional bias, stride, padding, and dilation.
/// Returns `true` if the kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_conv1d(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    input_layout: &Layout,
    weight_layout: &Layout,
    output_layout: &Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_numel: usize,
) -> bool {
    if !layouts_fit_cuda(&[input_layout, weight_layout, output_layout]) {
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
    let Some(mut out_n) = cuda_u32(out_numel) else {
        return false;
    };
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(func) = get_cuda_function("conv1d_f32") else {
        return false;
    };

    let Ok(in_layout_buf) = create_layout_buffer(input_layout) else {
        return false;
    };
    let Ok(w_layout_buf) = create_layout_buffer(weight_layout) else {
        return false;
    };
    let Ok(out_layout_buf) = create_layout_buffer(output_layout) else {
        return false;
    };

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut il_ptr = in_layout_buf.cu_deviceptr();
    let mut wl_ptr = w_layout_buf.cu_deviceptr();
    let mut ol_ptr = out_layout_buf.cu_deviceptr();

    let mut args: [*mut std::ffi::c_void; 11] = [
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut il_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut wl_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut ol_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut stride_val as *mut u32 as *mut std::ffi::c_void,
        &mut pad_val as *mut u32 as *mut std::ffi::c_void,
        &mut dil_val as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];

    let Some(grid_size) = launch_grid_size(out_numel) else {
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
        res == 0
    }
}

/// Launch the 2-D convolution kernel on the GPU.
///
/// Computes forward 2-D convolution with optional bias, stride, padding, and dilation.
/// Returns `true` if the kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_conv2d(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    input_layout: &Layout,
    weight_layout: &Layout,
    output_layout: &Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_numel: usize,
) -> bool {
    if !layouts_fit_cuda(&[input_layout, weight_layout, output_layout]) {
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
    let Some(mut out_n) = cuda_u32(out_numel) else {
        return false;
    };
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(func) = get_cuda_function("conv2d_f32") else {
        return false;
    };

    let Ok(in_layout_buf) = create_layout_buffer(input_layout) else {
        return false;
    };
    let Ok(w_layout_buf) = create_layout_buffer(weight_layout) else {
        return false;
    };
    let Ok(out_layout_buf) = create_layout_buffer(output_layout) else {
        return false;
    };

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut il_ptr = in_layout_buf.cu_deviceptr();
    let mut wl_ptr = w_layout_buf.cu_deviceptr();
    let mut ol_ptr = out_layout_buf.cu_deviceptr();

    let mut args: [*mut std::ffi::c_void; 11] = [
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut il_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut wl_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut ol_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut stride_val as *mut u32 as *mut std::ffi::c_void,
        &mut pad_val as *mut u32 as *mut std::ffi::c_void,
        &mut dil_val as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];

    let Some(grid_size) = launch_grid_size(out_numel) else {
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
        res == 0
    }
}

/// Launch the 3-D convolution kernel on the GPU.
///
/// Computes forward 3-D convolution with optional bias, stride, padding, and dilation.
/// Returns `true` if the kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_conv3d(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    input_layout: &Layout,
    weight_layout: &Layout,
    output_layout: &Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_numel: usize,
) -> bool {
    if !layouts_fit_cuda(&[input_layout, weight_layout, output_layout]) {
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
    let Some(mut out_n) = cuda_u32(out_numel) else {
        return false;
    };
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(func) = get_cuda_function("conv3d_f32") else {
        return false;
    };

    let Ok(in_layout_buf) = create_layout_buffer(input_layout) else {
        return false;
    };
    let Ok(w_layout_buf) = create_layout_buffer(weight_layout) else {
        return false;
    };
    let Ok(out_layout_buf) = create_layout_buffer(output_layout) else {
        return false;
    };

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut il_ptr = in_layout_buf.cu_deviceptr();
    let mut wl_ptr = w_layout_buf.cu_deviceptr();
    let mut ol_ptr = out_layout_buf.cu_deviceptr();

    let mut args: [*mut std::ffi::c_void; 11] = [
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut il_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut wl_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut ol_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut stride_val as *mut u32 as *mut std::ffi::c_void,
        &mut pad_val as *mut u32 as *mut std::ffi::c_void,
        &mut dil_val as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];

    let Some(grid_size) = launch_grid_size(out_numel) else {
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
        res == 0
    }
}
