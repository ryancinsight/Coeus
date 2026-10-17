//! Sliding-window extraction (unfold) and adjoint accumulation (fold).
//!
//! Thin const-rank dispatch over [`leto_ops`]: the per-axis window geometry
//! mirrors the [`coeus_ops`](https://github.com/ryancinsight/coeus)
//! `UnfoldFoldOps` surface one-to-one, so the CPU kernels delegate here
//! instead of re-implementing the traversal.

use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::{Result, WindowParameters};
use leto_ops::Scalar as LetoScalar;

/// Extract windows from an `[N, C, spatial...]` view into `[N, C*K, L]`.
///
/// Geometry (output locations, padding fill) is derived entirely by leto
/// from `parameters` and the input view; the output view must already have
/// the derived shape.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
fn unfold_n<T: LetoScalar, const R: usize, const D: usize>(
    input_layout: &CoeusLayout,
    input: &[T],
    kernel: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let input_view = to_leto_view::<T, R>(input_layout, input)?;
    let mut output_view = to_leto_view_mut::<T, 3>(output_layout, output)?;
    let parameters = WindowParameters::new(kernel, stride, padding, dilation)?;
    leto_ops::unfold_into::<T, R, D>(&input_view, parameters, &mut output_view)
}

/// Accumulate `[N, C*K, L]` columns into an `[N, C, spatial...]` view.
///
/// The output is zeroed before accumulation; overlapping windows sum,
/// matching the historical coeus CPU fold semantics.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
fn fold_n<T: LetoScalar, const R: usize, const D: usize>(
    input_layout: &CoeusLayout,
    input: &[T],
    output_spatial: [usize; D],
    kernel: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let input_view = to_leto_view::<T, 3>(input_layout, input)?;
    let mut output_view = to_leto_view_mut::<T, R>(output_layout, output)?;
    let parameters = WindowParameters::new(kernel, stride, padding, dilation)?;
    leto_ops::fold_into::<T, R, D>(&input_view, output_spatial, parameters, &mut output_view)
}

/// Extract one-dimensional sliding windows into caller-owned output.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_cpu::unfold1d_into;
///
/// let input_layout = Layout::new([1, 1, 5].into());
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let output_layout = Layout::new([1, 3, 3].into());
/// let mut output = [0.0_f64; 9];
/// unfold1d_into(&input_layout, &input, 3, 1, 0, 1, &output_layout, &mut output)
///     .unwrap();
/// assert_eq!(
///     output,
///     [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
/// );
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
pub fn unfold1d_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    unfold_n::<T, 3, 1>(
        input_layout,
        input,
        [kernel_size],
        [stride],
        [padding],
        [dilation],
        output_layout,
        output,
    )
}

/// Extract two-dimensional sliding windows into caller-owned output.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_cpu::unfold2d_into;
///
/// let input_layout = Layout::new([1, 1, 2, 2].into());
/// let input = [1.0_f64, 2.0, 3.0, 4.0];
/// let output_layout = Layout::new([1, 4, 1].into());
/// let mut output = [0.0_f64; 4];
/// unfold2d_into(
///     &input_layout, &input, 2, 2, 1, 1, 0, 0, 1, 1, &output_layout, &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
pub fn unfold2d_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    unfold_n::<T, 4, 2>(
        input_layout,
        input,
        [kernel_h, kernel_w],
        [stride_h, stride_w],
        [padding_h, padding_w],
        [dilation_h, dilation_w],
        output_layout,
        output,
    )
}

/// Accumulate one-dimensional windows into caller-owned output.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_cpu::fold1d_into;
///
/// let input_layout = Layout::new([1, 2, 3].into());
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let output_layout = Layout::new([1, 1, 4].into());
/// let mut output = [0.0_f64; 4];
/// fold1d_into(
///     &input_layout,
///     &input,
///     4,
///     2,
///     1,
///     0,
///     1,
///     &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [1.0, 6.0, 8.0, 6.0]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
pub fn fold1d_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    fold_n::<T, 3, 1>(
        input_layout,
        input,
        [output_size],
        [kernel_size],
        [stride],
        [padding],
        [dilation],
        output_layout,
        output,
    )
}

/// Accumulate two-dimensional windows into caller-owned output.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_cpu::fold2d_into;
///
/// let input_layout = Layout::new([1, 4, 4].into());
/// let input = [1.0_f64; 16];
/// let output_layout = Layout::new([1, 1, 3, 3].into());
/// let mut output = [0.0_f64; 9];
/// fold2d_into(
///     &input_layout, &input, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(
///     output,
///     [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
/// );
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps signature one-to-one"
)]
pub fn fold2d_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
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
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    fold_n::<T, 4, 2>(
        input_layout,
        input,
        [output_h, output_w],
        [kernel_h, kernel_w],
        [stride_h, stride_w],
        [padding_h, padding_w],
        [dilation_h, dilation_w],
        output_layout,
        output,
    )
}
