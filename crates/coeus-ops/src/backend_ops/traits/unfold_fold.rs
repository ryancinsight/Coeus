//! Unfold / Fold (im2col / col2im) sub-trait.
//!
//! [`UnfoldFoldOps`] is the interface-segregated sub-trait for sliding-window
//! extraction (unfold) and its adjoint accumulation (fold).  These are the
//! low-level building blocks for attention-patch and convolution-via-matmul
//! formulations, matching PyTorch `nn.Unfold` / `nn.Fold`.

use coeus_core::{ComputeBackend, Layout, Scalar};

/// Unfold / Fold operations.
///
/// This sub-trait is one of the interface-segregated concerns that compose
/// [`BackendOps`].  Backends implement `UnfoldFoldOps` directly; the blanket
/// impl in [`trait_def`] provides `BackendOps` automatically.
///
/// # Semantics
///
/// **Unfold 2D**: given `input` with layout `[N, C, H, W]` and kernel `(kH, kW)`,
/// extracts sliding windows into `output` with layout `[N, C*kH*kW, H_out*W_out]`.
/// Strides, padding, and dilation follow the PyTorch convention.
///
/// **Fold 2D**: inverse (adjoint) of unfold.  `input` layout `[N, C*kH*kW, L]`
/// accumulates into `output` layout `[N, C, H, W]`.  Multiple windows overlapping
/// the same output cell are summed.
///
/// 1D variants follow the same contract on `[N, C, L]` inputs.
///
/// [`BackendOps`]: super::super::BackendOps
/// [`trait_def`]: super::super::trait_def
pub trait UnfoldFoldOps<T: Scalar>: ComputeBackend {
    /// Unfold 1D: extract sliding windows from `[N, C, L]` into `[N, C*kernel, L_out]`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout, geometry, or dispatch
    /// validation fails.
    fn unfold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Fold 1D: accumulate `[N, C*kernel, L_out]` back into `[N, C, L]`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout, geometry, or dispatch
    /// validation fails.
    fn fold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Unfold 2D: extract sliding windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout, geometry, or dispatch
    /// validation fails.
    fn unfold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Fold 2D: accumulate `[N, C*kH*kW, L]` back into `[N, C, H, W]`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout, geometry, or dispatch
    /// validation fails.
    fn fold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
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
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
