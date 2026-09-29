// ── Pooling layer types (avg, max, global, adaptive) ──
//
// Organized into sub-modules to keep each variant-focused.

mod adaptive;
mod avg;
mod global;
mod max;
mod pool1d;

use crate::module::ModuleError;
use std::error::Error;

pub use adaptive::{AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d};
pub use avg::{AvgPool2d, AvgPool3d};
pub use global::{
    GlobalAvgPool1d, GlobalAvgPool2d, GlobalAvgPool3d, GlobalMaxPool2d, GlobalMaxPool3d,
};
pub use max::{MaxPool2d, MaxPool3d};
pub use pool1d::{AvgPool1d, MaxPool1d};

// ── Shared pooling helpers ──

#[inline]
pub(crate) fn k_eff(kernel_size: usize, dilation: usize) -> Option<usize> {
    kernel_size
        .checked_sub(1)
        .and_then(|extent| dilation.checked_mul(extent))
        .and_then(|extent| extent.checked_add(1))
}

#[inline]
pub(crate) fn out_dim(
    input_dim: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return None;
    }

    let padded = padding
        .checked_mul(2)
        .and_then(|padding| input_dim.checked_add(padding))?;
    let numerator = padded.checked_sub(k_eff(kernel_size, dilation)?)?;
    numerator
        .checked_div(stride)
        .and_then(|quotient| quotient.checked_add(1))
}

#[inline]
pub(crate) fn checked_out_dim<E>(
    module: &'static str,
    input_dim: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<usize, ModuleError<E>>
where
    E: Error + 'static,
{
    out_dim(input_dim, kernel_size, padding, stride, dilation)
        .filter(|&output| output != 0)
        .ok_or_else(|| ModuleError::ShapeMismatch {
            module,
            parameter: "pooling window",
            expected: vec![1],
            actual: vec![input_dim, kernel_size, padding, stride, dilation],
        })
}
