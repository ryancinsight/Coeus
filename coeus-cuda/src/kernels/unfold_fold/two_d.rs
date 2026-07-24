use super::dispatch::{kernel, launch};
use super::validation::{checked_2d_launch, checked_output_dim, Launch2d, Parameters2d};
use crate::backend::CudaScalar;
use crate::driver::CUdeviceptr;
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::Layout;

/// Dispatch two-dimensional sliding-window extraction.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold2d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let [n, channels, input_h, input_w] = match input_layout.shape() {
        [n, channels, input_h, input_w] => [*n, *channels, *input_h, *input_w],
        _ => return false,
    };
    let Some(output_h) = checked_output_dim(input_h, kernel_h, padding_h, stride_h, dilation_h)
    else {
        return false;
    };
    let Some(output_w) = checked_output_dim(input_w, kernel_w, padding_w, stride_w, dilation_w)
    else {
        return false;
    };
    let Some(kernel_area) = kernel_h.checked_mul(kernel_w) else {
        return false;
    };
    let Some(output_locations) = output_h.checked_mul(output_w) else {
        return false;
    };
    let Some(output_channels) = channels.checked_mul(kernel_area) else {
        return false;
    };
    if output_layout.shape() != [n, output_channels, output_locations] {
        return false;
    }
    let Some(dimensions) = checked_2d_launch(
        input,
        input_layout,
        output,
        output_layout,
        Parameters2d {
            input_rank: 4,
            output_rank: 3,
            values: [
                kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h,
                dilation_w, output_w,
            ],
            width: output_w,
        },
    ) else {
        return false;
    };
    dispatch_unfold_or_fold2d(
        "unfold2d_kernel",
        input,
        input_layout,
        dimensions,
        output,
        output_layout,
    )
}

/// Dispatch two-dimensional adjoint fold accumulation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold2d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let [n, input_channels, input_locations] = match input_layout.shape() {
        [n, input_channels, input_locations] => [*n, *input_channels, *input_locations],
        _ => return false,
    };
    let [output_n, channels, output_h, output_w] = match output_layout.shape() {
        [output_n, channels, output_h, output_w] => [*output_n, *channels, *output_h, *output_w],
        _ => return false,
    };
    if n != output_n {
        return false;
    }
    let Some(kernel_area) = kernel_h.checked_mul(kernel_w) else {
        return false;
    };
    let Some(expected_input_h) =
        checked_output_dim(output_h, kernel_h, padding_h, stride_h, dilation_h)
    else {
        return false;
    };
    let Some(input_w) = checked_output_dim(output_w, kernel_w, padding_w, stride_w, dilation_w)
    else {
        return false;
    };
    let Some(expected_input_locations) = expected_input_h.checked_mul(input_w) else {
        return false;
    };
    let Some(expected_input_channels) = channels.checked_mul(kernel_area) else {
        return false;
    };
    if input_channels != expected_input_channels || input_locations != expected_input_locations {
        return false;
    }
    let Some(dimensions) = checked_2d_launch(
        input,
        input_layout,
        output,
        output_layout,
        Parameters2d {
            input_rank: 3,
            output_rank: 4,
            values: [
                kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h,
                dilation_w, input_w,
            ],
            width: input_w,
        },
    ) else {
        return false;
    };
    dispatch_unfold_or_fold2d(
        "fold2d_kernel",
        input,
        input_layout,
        dimensions,
        output,
        output_layout,
    )
}

fn dispatch_unfold_or_fold2d<T: CudaScalar>(
    name: &str,
    input: &CudaStorage<T>,
    input_layout: &Layout,
    dimensions: Launch2d,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(compiled) = kernel::<T>(name) else {
        return false;
    };
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let Ok(input_gpu) = GpuLayoutInfo::try_from(input_layout) else {
        return false;
    };
    let Ok(output_gpu) = GpuLayoutInfo::try_from(output_layout) else {
        return false;
    };
    let mut values = dimensions.values;
    let total = dimensions.total_elements;
    let mut total_gpu = dimensions.total;
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut values[0] as *mut u32 as *mut _,
        &mut values[1] as *mut u32 as *mut _,
        &mut values[2] as *mut u32 as *mut _,
        &mut values[3] as *mut u32 as *mut _,
        &mut values[4] as *mut u32 as *mut _,
        &mut values[5] as *mut u32 as *mut _,
        &mut values[6] as *mut u32 as *mut _,
        &mut values[7] as *mut u32 as *mut _,
        &mut values[8] as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch(name, total, compiled.func, &mut args)
}
