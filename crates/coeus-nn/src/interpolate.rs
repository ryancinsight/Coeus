// ── Interpolate — spatial resize / upsample / downsample ──
//
// Supported modes:
//   `nearest`  — map each output pixel to the nearest input pixel (floor)
//   `bilinear` — bilinear interpolation (2-neighbour per axis; linear in 1-D)
//   `bicubic`  — cubic convolution, a = -0.75 (4-neighbour per axis)
//   `area`     — average of every source element inside the output's box
//
// Input conventions (matching PyTorch and Burn):
//   1-D: `[N, C, L]`         → output `[N, C, new_L]`
//   2-D: `[N, C, H, W]`      → output `[N, C, new_H, new_W]`
//
// The resize kernels live in `coeus_ops::resize`, together with their
// reverse-mode scatter, so the differentiable layer (and the Python binding)
// differentiate the same taps this forward pass applies. This module is the
// raw-tensor face of those kernels; use `coeus_autograd`'s `resize_1d` /
// `resize_2d` when the input is a tracked `Var`.

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
pub use coeus_ops::resize::ResizeMode as InterpolateMode;
use coeus_tensor::Tensor;

/// Resize a 1-D spatial tensor `[N, C, L]` to `[N, C, new_L]`.
///
/// Supports every [`InterpolateMode`].
pub fn interpolate_1d<T: Float + FloatElement, B: coeus_core::Backend>(
    input: &Tensor<T, B>,
    new_l: usize,
    mode: InterpolateMode,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_ops::resize_1d(input, new_l, mode)
}

/// Resize a 2-D spatial tensor `[N, C, H, W]` to `[N, C, new_H, new_W]`.
///
/// Supports every [`InterpolateMode`].
pub fn interpolate_2d<T: Float + FloatElement, B: coeus_core::Backend>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    mode: InterpolateMode,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_ops::resize_2d(input, new_h, new_w, mode)
}
