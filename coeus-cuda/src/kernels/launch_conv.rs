#![allow(clippy::too_many_arguments)]

use super::{create_layout_buffer, get_cuda_function};
use crate::driver::CudaDriver;
use crate::storage::CudaStorage;
use coeus_core::Layout;

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
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(func) = get_cuda_function("conv1d_f32") else {
        return false;
    };

    let in_layout_buf = create_layout_buffer(input_layout);
    let w_layout_buf = create_layout_buffer(weight_layout);
    let out_layout_buf = create_layout_buffer(output_layout);

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut il_ptr = in_layout_buf.cu_deviceptr();
    let mut wl_ptr = w_layout_buf.cu_deviceptr();
    let mut ol_ptr = out_layout_buf.cu_deviceptr();
    let mut stride_val = stride as u32;
    let mut pad_val = padding as u32;
    let mut dil_val = dilation as u32;
    let mut out_n = out_numel as u32;

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

    let block_size = 256;
    let grid_size = out_numel.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            func,
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
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(func) = get_cuda_function("conv2d_f32") else {
        return false;
    };

    let in_layout_buf = create_layout_buffer(input_layout);
    let w_layout_buf = create_layout_buffer(weight_layout);
    let out_layout_buf = create_layout_buffer(output_layout);

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut il_ptr = in_layout_buf.cu_deviceptr();
    let mut wl_ptr = w_layout_buf.cu_deviceptr();
    let mut ol_ptr = out_layout_buf.cu_deviceptr();
    let mut stride_val = stride as u32;
    let mut pad_val = padding as u32;
    let mut dil_val = dilation as u32;
    let mut out_n = out_numel as u32;

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

    let block_size = 256;
    let grid_size = out_numel.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            func,
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

pub fn launch_conv1d_backward(
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
    let drv = match CudaDriver::get() {
        Some(d) => d,
        None => return false,
    };

    let go_layout_buf = create_layout_buffer(grad_out_layout);
    let mut go_ptr = grad_out.cu_deviceptr();
    let mut gol_ptr = go_layout_buf.cu_deviceptr();

    let mut stride_val = stride as u32;
    let mut pad_val = padding as u32;
    let mut dil_val = dilation as u32;

    if let Some(gi) = grad_input {
        let Some(func) = get_cuda_function("conv1d_grad_input_f32") else {
            return false;
        };
        let w_layout_buf = create_layout_buffer(weight_layout);
        let gi_layout_buf = create_layout_buffer(grad_input_layout);

        let mut w_ptr = weight.cu_deviceptr();
        let mut gi_ptr = gi.cu_deviceptr();
        let mut wl_ptr = w_layout_buf.cu_deviceptr();
        let mut gil_ptr = gi_layout_buf.cu_deviceptr();
        let numel_in = grad_input_layout.shape().iter().product::<usize>();
        let mut num_in = numel_in as u32;

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

        let block_size = 256;
        let grid_size = numel_in.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gw) = grad_weight {
        let Some(func) = get_cuda_function("conv1d_grad_weight_f32") else {
            return false;
        };
        let in_layout_buf = create_layout_buffer(input_layout);
        let gw_layout_buf = create_layout_buffer(grad_weight_layout);

        let mut in_ptr = input.cu_deviceptr();
        let mut gw_ptr = gw.cu_deviceptr();
        let mut inl_ptr = in_layout_buf.cu_deviceptr();
        let mut gwl_ptr = gw_layout_buf.cu_deviceptr();
        let numel_w = grad_weight_layout.shape().iter().product::<usize>();
        let mut num_w = numel_w as u32;

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

        let block_size = 256;
        let grid_size = numel_w.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gb) = grad_bias {
        let Some(func) = get_cuda_function("conv1d_grad_bias_f32") else {
            return false;
        };
        let mut gb_ptr = gb.cu_deviceptr();
        let c_out = weight_layout.shape()[0];
        let mut c_out_val = c_out as u32;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gb_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gol_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut c_out_val as *mut u32 as *mut std::ffi::c_void,
        ];

        let block_size = 256;
        let grid_size = c_out.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    true
}

pub fn launch_conv2d_backward(
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
    let drv = match CudaDriver::get() {
        Some(d) => d,
        None => return false,
    };

    let go_layout_buf = create_layout_buffer(grad_out_layout);
    let mut go_ptr = grad_out.cu_deviceptr();
    let mut gol_ptr = go_layout_buf.cu_deviceptr();

    let mut stride_val = stride as u32;
    let mut pad_val = padding as u32;
    let mut dil_val = dilation as u32;

    if let Some(gi) = grad_input {
        let Some(func) = get_cuda_function("conv2d_grad_input_f32") else {
            return false;
        };
        let w_layout_buf = create_layout_buffer(weight_layout);
        let gi_layout_buf = create_layout_buffer(grad_input_layout);

        let mut w_ptr = weight.cu_deviceptr();
        let mut gi_ptr = gi.cu_deviceptr();
        let mut wl_ptr = w_layout_buf.cu_deviceptr();
        let mut gil_ptr = gi_layout_buf.cu_deviceptr();
        let numel_in = grad_input_layout.shape().iter().product::<usize>();
        let mut num_in = numel_in as u32;

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

        let block_size = 256;
        let grid_size = numel_in.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gw) = grad_weight {
        let Some(func) = get_cuda_function("conv2d_grad_weight_f32") else {
            return false;
        };
        let in_layout_buf = create_layout_buffer(input_layout);
        let gw_layout_buf = create_layout_buffer(grad_weight_layout);

        let mut in_ptr = input.cu_deviceptr();
        let mut gw_ptr = gw.cu_deviceptr();
        let mut inl_ptr = in_layout_buf.cu_deviceptr();
        let mut gwl_ptr = gw_layout_buf.cu_deviceptr();
        let numel_w = grad_weight_layout.shape().iter().product::<usize>();
        let mut num_w = numel_w as u32;

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

        let block_size = 256;
        let grid_size = numel_w.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    if let Some(gb) = grad_bias {
        let Some(func) = get_cuda_function("conv2d_grad_bias_f32") else {
            return false;
        };
        let mut gb_ptr = gb.cu_deviceptr();
        let c_out = weight_layout.shape()[0];
        let mut c_out_val = c_out as u32;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gb_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gol_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut c_out_val as *mut u32 as *mut std::ffi::c_void,
        ];

        let block_size = 256;
        let grid_size = c_out.div_ceil(block_size);
        unsafe {
            let res = (drv.cu_launch_kernel)(
                func,
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
            if res != 0 {
                return false;
            }
        }
    }

    true
}
