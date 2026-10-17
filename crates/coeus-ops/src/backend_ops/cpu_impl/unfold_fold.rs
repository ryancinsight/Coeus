//! Sliding-window extraction (unfold) and adjoint accumulation (fold) over CPU
//! storage, delegated to the leto kernels.
//!
//! Geometry, shape, and storage validation stays here ([`validation`]):
//! leto accepts kernel-exceeds-input as an empty success and admits
//! strided-injective outputs, while this surface rejects both (matching
//! PyTorch and the historical CPU contract). The traversal itself — window
//! decomposition, padded coordinates, fold accumulation order — is proven
//! identical to [`coeus_cpu`], so delegation is value-preserving.

use super::error::map_leto_error;
use coeus_core::{BackendError, Layout};
use leto_ops::Scalar as LetoScalar;

#[cfg(test)]
mod tests;
pub(super) mod validation;

#[expect(
    clippy::too_many_arguments,
    reason = "delegation mirrors the UnfoldFoldOps signature one-to-one"
)]
pub(super) fn unfold1d<T: LetoScalar>(
    input: &[T],
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    coeus_cpu::unfold1d_into(
        input_layout,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output,
    )
    .map_err(|error| map_leto_error("unfold1d", error))
}

#[expect(
    clippy::too_many_arguments,
    reason = "delegation mirrors the UnfoldFoldOps signature one-to-one"
)]
pub(super) fn fold1d<T: LetoScalar>(
    input: &[T],
    input_layout: &Layout,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    coeus_cpu::fold1d_into(
        input_layout,
        input,
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output,
    )
    .map_err(|error| map_leto_error("fold1d", error))
}

#[expect(
    clippy::too_many_arguments,
    reason = "delegation mirrors the UnfoldFoldOps signature one-to-one"
)]
pub(super) fn unfold2d<T: LetoScalar>(
    input: &[T],
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    coeus_cpu::unfold2d_into(
        input_layout,
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_layout,
        output,
    )
    .map_err(|error| map_leto_error("unfold2d", error))
}

#[expect(
    clippy::too_many_arguments,
    reason = "delegation mirrors the UnfoldFoldOps signature one-to-one"
)]
pub(super) fn fold2d<T: LetoScalar>(
    input: &[T],
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
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    coeus_cpu::fold2d_into(
        input_layout,
        input,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_layout,
        output,
    )
    .map_err(|error| map_leto_error("fold2d", error))
}
