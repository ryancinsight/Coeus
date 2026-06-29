// ── Pooling layer types (avg, max, global) ──
//
// Organized into sub-modules to keep each variant-focused.

mod avg;
mod global;
mod max;
mod pool1d;

pub use avg::{AvgPool2d, AvgPool3d};
pub use global::{
    GlobalAvgPool1d, GlobalAvgPool2d, GlobalAvgPool3d, GlobalMaxPool2d, GlobalMaxPool3d,
};
pub use max::{MaxPool2d, MaxPool3d};
pub use pool1d::{AvgPool1d, MaxPool1d};

// ── Shared pooling helpers ──

#[inline]
pub(crate) fn k_eff(kernel_size: usize, dilation: usize) -> usize {
    dilation * (kernel_size - 1) + 1
}

#[inline]
pub(crate) fn out_dim(
    input_dim: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let total = input_dim + 2 * padding;
    match total.checked_sub(k_eff(kernel_size, dilation)) {
        Some(numer) => numer / stride + 1,
        None => 0,
    }
}
