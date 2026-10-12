use super::shape_n;
use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::{LetoError, Result, WindowParameters};
use leto_ops::Scalar as LetoScalar;

/// Unfold over const-rank leto views.
///
/// `[N, C, spatial...]` becomes `[N, C*K, L]`, where `K` is the kernel volume
/// and `L` is the derived window-location count.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps unfold signature one-to-one"
)]
fn unfold_n<T: LetoScalar, const R: usize, const D: usize>(
    input_layout: &CoeusLayout,
    input: &[T],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let input_view = to_leto_view::<T, R>(input_layout, input)?;
    let mut output_view = to_leto_view_mut::<T, 3>(output_layout, output)?;
    let parameters = WindowParameters::new(
        shape_n::<D>(kernel)?,
        shape_n::<D>(stride)?,
        shape_n::<D>(padding)?,
        shape_n::<D>(dilation)?,
    )?;
    leto_ops::unfold_into(&input_view, parameters, &mut output_view)
}

/// Fold over const-rank leto views.
///
/// The output is zeroed before accumulation, so overlapping windows implement
/// the adjoint of [`unfold_into`] with deterministic serial writes.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps fold signature one-to-one"
)]
fn fold_n<T: LetoScalar, const R: usize, const D: usize>(
    input_layout: &CoeusLayout,
    input: &[T],
    output_spatial: [usize; D],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let input_view = to_leto_view::<T, 3>(input_layout, input)?;
    let mut output_view = to_leto_view_mut::<T, R>(output_layout, output)?;
    let parameters = WindowParameters::new(
        shape_n::<D>(kernel)?,
        shape_n::<D>(stride)?,
        shape_n::<D>(padding)?,
        shape_n::<D>(dilation)?,
    )?;
    leto_ops::fold_into(&input_view, output_spatial, parameters, &mut output_view)
}

/// Extract sliding windows of a `[N, C, spatial...]` tensor into `[N, C*K, L]`.
///
/// The input rank selects the monomorphized leto kernel: rank 3 unfolds over
/// one spatial axis, rank 4 over two. Window parameters arrive per spatial
/// axis, matching the [`coeus_ops`](https://github.com/ryancinsight/coeus)
/// `UnfoldFoldOps` surface.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::unfold_into;
///
/// let input_layout = Layout::new([1, 1, 4].into());
/// let input = [1_i32, 2, 3, 4];
/// let output_layout = Layout::new([1, 2, 3].into());
/// let mut output = [0_i32; 6];
/// unfold_into(
///     &input_layout,
///     &input,
///     &[2],
///     &[1],
///     &[0],
///     &[1],
///     &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [1, 2, 3, 2, 3, 4]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps unfold signature one-to-one"
)]
pub fn unfold_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    match input_layout.ndim() {
        3 => unfold_n::<T, 3, 1>(
            input_layout,
            input,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        4 => unfold_n::<T, 4, 2>(
            input_layout,
            input,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        rank => Err(LetoError::StorageError {
            reason: format!("unfold expects rank 3..=4 ([N, C, spatial...]), got {rank}"),
        }),
    }
}

/// Accumulate `[N, C*K, L]` columns back into a `[N, C, spatial...]` tensor.
///
/// `output_spatial` names the spatial extents being reconstructed; the output
/// rank selects the monomorphized leto kernel.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::fold_into;
///
/// let input_layout = Layout::new([1, 2, 3].into());
/// let input = [1_i32, 2, 3, 2, 3, 4];
/// let output_layout = Layout::new([1, 1, 4].into());
/// let mut output = [0_i32; 4];
/// fold_into(
///     &input_layout,
///     &input,
///     &[4],
///     &[2],
///     &[1],
///     &[0],
///     &[1],
///     &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [1, 4, 6, 4]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus UnfoldFoldOps fold signature one-to-one"
)]
pub fn fold_into<T: LetoScalar>(
    input_layout: &CoeusLayout,
    input: &[T],
    output_spatial: &[usize],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    match output_layout.ndim() {
        3 => fold_n::<T, 3, 1>(
            input_layout,
            input,
            shape_n::<1>(output_spatial)?,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        4 => fold_n::<T, 4, 2>(
            input_layout,
            input,
            shape_n::<2>(output_spatial)?,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        rank => Err(LetoError::StorageError {
            reason: format!("fold expects rank 3..=4 ([N, C, spatial...]), got {rank}"),
        }),
    }
}
