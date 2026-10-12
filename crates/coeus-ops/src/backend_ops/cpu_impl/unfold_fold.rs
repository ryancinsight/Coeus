// ── Unfold / Fold CPU kernel delegations (1D and 2D) ──
//
// Unfold 1D: [N, C, L] → [N, C*kernel_size, L_out]
// Fold   1D: [N, C*kernel_size, L_out] → [N, C, L]  (accumulate/sum overlap)
//
// Unfold 2D: [N, C, H, W] → [N, C*kH*kW, H_out*W_out]
// Fold   2D: [N, C*kH*kW, H_out*W_out] → [N, C, H, W]
//
// All kernels follow the PyTorch nn.Unfold / nn.Fold convention. Window
// geometry, shape validation, and traversal live in leto-ops; these functions
// only adapt coeus storage to the coeus-leto dispatch boundary.
#![allow(clippy::too_many_arguments)]

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::error::map_leto_error;
use super::CpuBackend;

/// Extract sliding windows from `[N, C, L]` into `[N, C*kernel_size, L_out]`.
#[inline]
pub(crate) fn unfold1d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_leto::unfold_into(
        input_layout,
        input.as_slice(),
        &[kernel_size],
        &[stride],
        &[padding],
        &[dilation],
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("unfold 1d", error))
}

/// Accumulate `[N, C*kernel_size, L_out]` back into `[N, C, L]`.
#[inline]
pub(crate) fn fold1d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_leto::fold_into(
        input_layout,
        input.as_slice(),
        &[output_size],
        &[kernel_size],
        &[stride],
        &[padding],
        &[dilation],
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("fold 1d", error))
}

/// Extract sliding windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
#[inline]
pub(crate) fn unfold2d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_leto::unfold_into(
        input_layout,
        input.as_slice(),
        &[kernel_h, kernel_w],
        &[stride_h, stride_w],
        &[padding_h, padding_w],
        &[dilation_h, dilation_w],
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("unfold 2d", error))
}

/// Accumulate `[N, C*kH*kW, H_out*W_out]` back into `[N, C, output_h, output_w]`.
#[inline]
pub(crate) fn fold2d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
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
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    coeus_leto::fold_into(
        input_layout,
        input.as_slice(),
        &[output_h, output_w],
        &[kernel_h, kernel_w],
        &[stride_h, stride_w],
        &[padding_h, padding_w],
        &[dilation_h, dilation_w],
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("fold 2d", error))
}
