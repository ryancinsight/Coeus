use super::dispatch::{kernel, launch};
use super::validation::{Parameters1d, checked_1d_launch, checked_output_dim};
use crate::backend::CudaScalar;
use crate::driver::CUdeviceptr;
use crate::kernels::GpuLayoutInfo;
use crate::storage::CudaStorage;
use coeus_core::Layout;

/// Dispatch one-dimensional sliding-window extraction.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(dimensions) = checked_1d_launch(
        input,
        input_layout,
        output,
        output_layout,
        Parameters1d {
            kernel_size,
            stride,
            padding,
            dilation,
        },
    ) else {
        return false;
    };
    let [n, channels, input_length] = input_layout.shape() else {
        return false;
    };
    let Some(output_length) =
        checked_output_dim(*input_length, kernel_size, padding, stride, dilation)
    else {
        return false;
    };
    let Some(output_channels) = channels.checked_mul(kernel_size) else {
        return false;
    };
    if output_layout.shape() != [*n, output_channels, output_length] {
        return false;
    }
    let Some(compiled) = kernel::<T>("unfold1d_kernel") else {
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
    let mut kernel_size = dimensions.kernel_size;
    let mut stride = dimensions.stride;
    let mut padding = dimensions.padding;
    let mut dilation = dimensions.dilation;
    let total = dimensions.total_elements;
    let mut total_gpu = dimensions.total;
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut kernel_size as *mut u32 as *mut _,
        &mut stride as *mut u32 as *mut _,
        &mut padding as *mut u32 as *mut _,
        &mut dilation as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch("unfold1d", total, compiled.func, &mut args)
}

/// Dispatch one-dimensional adjoint fold accumulation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    dispatch_unfold_or_fold1d(
        "fold1d_kernel",
        input,
        input_layout,
        Parameters1d {
            kernel_size,
            stride,
            padding,
            dilation,
        },
        output,
        output_layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_unfold_or_fold1d<T: CudaScalar>(
    name: &str,
    input: &CudaStorage<T>,
    input_layout: &Layout,
    parameters: Parameters1d,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(dimensions) =
        checked_1d_launch(input, input_layout, output, output_layout, parameters)
    else {
        return false;
    };
    let Parameters1d {
        kernel_size,
        stride,
        padding,
        dilation,
    } = parameters;
    let [n, input_channels, input_length] = input_layout.shape() else {
        return false;
    };
    let input_channels = *input_channels;
    let output_shape = output_layout.shape();
    let Some(channels) = input_channels
        .checked_div(kernel_size)
        .filter(|_| input_channels.is_multiple_of(kernel_size))
    else {
        return false;
    };
    let output_length = output_shape[2];
    let Some(expected_input_length) =
        checked_output_dim(output_length, kernel_size, padding, stride, dilation)
    else {
        return false;
    };
    if input_layout.shape() != [*n, input_channels, expected_input_length]
        || output_shape != [*n, channels, output_length]
        || *input_length != expected_input_length
    {
        return false;
    }
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
    let mut kernel_size = dimensions.kernel_size;
    let mut stride = dimensions.stride;
    let mut padding = dimensions.padding;
    let mut dilation = dimensions.dilation;
    let total = dimensions.total_elements;
    let mut total_gpu = dimensions.total;
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut kernel_size as *mut u32 as *mut _,
        &mut stride as *mut u32 as *mut _,
        &mut padding as *mut u32 as *mut _,
        &mut dilation as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch(name, total, compiled.func, &mut args)
}
