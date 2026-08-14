use super::validation::{
    checked_product, output_width, parameter, require_dimension, require_rank,
    require_signed_coordinates,
};
use super::{dispatch, KernelKind};
use crate::backend::WgpuScalar;
use coeus_core::Layout;

/// Dispatch one-dimensional sliding-window extraction on the device.
///
/// # Errors
///
/// Returns a typed validation error before device access when the layouts,
/// window geometry, WGSL parameters, or dispatch count are not representable.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_unfold1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError> {
    const OPERATION: &str = "unfold1d";
    require_rank(OPERATION, input_layout, 3)?;
    require_rank(OPERATION, output_layout, 3)?;
    let width = input_layout.shape()[2];
    let window_count = output_width(OPERATION, width, kernel_size, stride, padding, dilation)?;
    let output_channels = checked_product(
        OPERATION,
        "output-channel arithmetic overflow",
        &[input_layout.shape()[1], kernel_size],
    )?;
    require_dimension(OPERATION, output_layout, 0, input_layout.shape()[0])?;
    require_dimension(OPERATION, output_layout, 1, output_channels)?;
    require_dimension(OPERATION, output_layout, 2, window_count)?;
    require_signed_coordinates(
        OPERATION,
        window_count,
        kernel_size,
        stride,
        padding,
        dilation,
    )?;
    let params = [
        parameter(OPERATION, kernel_size, "kernel_size", true)?,
        parameter(OPERATION, stride, "stride", true)?,
        parameter(OPERATION, padding, "padding", false)?,
        parameter(OPERATION, dilation, "dilation", true)?,
        0,
        0,
        0,
        0,
        parameter(OPERATION, window_count, "output_width", false)?,
    ];
    dispatch::<T>(
        KernelKind::Unfold1d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    )
}

/// Dispatch one-dimensional adjoint fold accumulation on the device.
///
/// # Errors
///
/// Returns a typed validation error before device access when the layouts,
/// window geometry, WGSL parameters, or dispatch count are not representable.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_fold1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError> {
    const OPERATION: &str = "fold1d";
    require_rank(OPERATION, input_layout, 3)?;
    require_rank(OPERATION, output_layout, 3)?;
    require_dimension(OPERATION, output_layout, 2, output_size)?;
    let input_channels = checked_product(
        OPERATION,
        "input-channel arithmetic overflow",
        &[output_layout.shape()[1], kernel_size],
    )?;
    let window_count = output_width(
        OPERATION,
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
    )?;
    require_dimension(OPERATION, input_layout, 0, output_layout.shape()[0])?;
    require_dimension(OPERATION, input_layout, 1, input_channels)?;
    require_dimension(OPERATION, input_layout, 2, window_count)?;
    require_signed_coordinates(
        OPERATION,
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
    )?;
    let params = [
        parameter(OPERATION, kernel_size, "kernel_size", true)?,
        parameter(OPERATION, stride, "stride", true)?,
        parameter(OPERATION, padding, "padding", false)?,
        parameter(OPERATION, dilation, "dilation", true)?,
        0,
        0,
        0,
        0,
        0,
    ];
    dispatch::<T>(
        KernelKind::Fold1d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    )
}

/// Dispatch two-dimensional sliding-window extraction on the device.
///
/// # Errors
///
/// Returns a typed validation error before device access when the layouts,
/// window geometry, WGSL parameters, or dispatch count are not representable.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_unfold2d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError> {
    const OPERATION: &str = "unfold2d";
    require_rank(OPERATION, input_layout, 4)?;
    require_rank(OPERATION, output_layout, 3)?;
    let height = input_layout.shape()[2];
    let width = input_layout.shape()[3];
    let output_height = output_width(OPERATION, height, kernel_h, stride_h, padding_h, dilation_h)?;
    let output_width = output_width(OPERATION, width, kernel_w, stride_w, padding_w, dilation_w)?;
    let output_channels = checked_product(
        OPERATION,
        "output-channel arithmetic overflow",
        &[input_layout.shape()[1], kernel_h, kernel_w],
    )?;
    let output_locations = checked_product(
        OPERATION,
        "output-location arithmetic overflow",
        &[output_height, output_width],
    )?;
    require_dimension(OPERATION, output_layout, 0, input_layout.shape()[0])?;
    require_dimension(OPERATION, output_layout, 1, output_channels)?;
    require_dimension(OPERATION, output_layout, 2, output_locations)?;
    require_signed_coordinates(
        OPERATION,
        output_height,
        kernel_h,
        stride_h,
        padding_h,
        dilation_h,
    )?;
    require_signed_coordinates(
        OPERATION,
        output_width,
        kernel_w,
        stride_w,
        padding_w,
        dilation_w,
    )?;
    let params = [
        parameter(OPERATION, kernel_h, "kernel_h", true)?,
        parameter(OPERATION, kernel_w, "kernel_w", true)?,
        parameter(OPERATION, stride_h, "stride_h", true)?,
        parameter(OPERATION, stride_w, "stride_w", true)?,
        parameter(OPERATION, padding_h, "padding_h", false)?,
        parameter(OPERATION, padding_w, "padding_w", false)?,
        parameter(OPERATION, dilation_h, "dilation_h", true)?,
        parameter(OPERATION, dilation_w, "dilation_w", true)?,
        parameter(OPERATION, output_width, "output_width", false)?,
    ];
    dispatch::<T>(
        KernelKind::Unfold2d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    )
}

/// Dispatch two-dimensional adjoint fold accumulation on the device.
///
/// # Errors
///
/// Returns a typed validation error before device access when the layouts,
/// window geometry, WGSL parameters, or dispatch count are not representable.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_fold2d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError> {
    const OPERATION: &str = "fold2d";
    require_rank(OPERATION, output_layout, 4)?;
    require_dimension(OPERATION, output_layout, 2, output_h)?;
    require_dimension(OPERATION, output_layout, 3, output_w)?;
    require_rank(OPERATION, input_layout, 3)?;
    let input_channels = checked_product(
        OPERATION,
        "input-channel arithmetic overflow",
        &[output_layout.shape()[1], kernel_h, kernel_w],
    )?;
    let input_height = output_width(
        OPERATION, output_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    let input_width = output_width(
        OPERATION, output_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    let input_locations = checked_product(
        OPERATION,
        "input-location arithmetic overflow",
        &[input_height, input_width],
    )?;
    require_dimension(OPERATION, input_layout, 0, output_layout.shape()[0])?;
    require_dimension(OPERATION, input_layout, 1, input_channels)?;
    require_dimension(OPERATION, input_layout, 2, input_locations)?;
    require_signed_coordinates(
        OPERATION, output_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    require_signed_coordinates(
        OPERATION, output_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    let params = [
        parameter(OPERATION, kernel_h, "kernel_h", true)?,
        parameter(OPERATION, kernel_w, "kernel_w", true)?,
        parameter(OPERATION, stride_h, "stride_h", true)?,
        parameter(OPERATION, stride_w, "stride_w", true)?,
        parameter(OPERATION, padding_h, "padding_h", false)?,
        parameter(OPERATION, padding_w, "padding_w", false)?,
        parameter(OPERATION, dilation_h, "dilation_h", true)?,
        parameter(OPERATION, dilation_w, "dilation_w", true)?,
        parameter(OPERATION, input_width, "output_width", false)?,
    ];
    dispatch::<T>(
        KernelKind::Fold2d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    )
}
