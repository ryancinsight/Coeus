use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::{LetoError, Result, WindowParameters};
use leto_ops::{PoolingMode, Scalar as LetoScalar};

/// Forward pooling over const-rank leto views.
///
/// Scalar window parameters broadcast over every spatial axis, matching the
/// [`coeus_ops`](https://github.com/ryancinsight/coeus) `PoolOps` surface.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus PoolOps forward signature one-to-one"
)]
fn pooling_forward_n<T: LetoScalar, const R: usize, const D: usize>(
    mode: PoolingMode,
    input_layout: &CoeusLayout,
    input: &[T],
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let input_view = to_leto_view::<T, R>(input_layout, input)?;
    let mut output_view = to_leto_view_mut::<T, R>(output_layout, output)?;
    let parameters = WindowParameters::new([kernel; D], [stride; D], [padding; D], [dilation; D])?;
    leto_ops::pooling_forward_into(&input_view, parameters, mode, &mut output_view)
}

/// Backward pooling over const-rank leto views.
///
/// `input` carries the forward input for [`PoolingMode::Maximum`]; the average
/// path derives its geometry from `grad_input` and passes `None`.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus PoolOps backward signature one-to-one"
)]
fn pooling_backward_n<T: LetoScalar, const R: usize, const D: usize>(
    mode: PoolingMode,
    grad_output_layout: &CoeusLayout,
    grad_output: &[T],
    input: Option<(&CoeusLayout, &[T])>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input_layout: &CoeusLayout,
    grad_input: &mut [T],
) -> Result<()> {
    let grad_output_view = to_leto_view::<T, R>(grad_output_layout, grad_output)?;
    let input_view = input
        .map(|(layout, data)| to_leto_view::<T, R>(layout, data))
        .transpose()?;
    let mut grad_input_view = to_leto_view_mut::<T, R>(grad_input_layout, grad_input)?;
    let parameters = WindowParameters::new([kernel; D], [stride; D], [padding; D], [dilation; D])?;
    leto_ops::pooling_backward_accumulate(
        &grad_output_view,
        input_view.as_ref(),
        parameters,
        mode,
        &mut grad_input_view,
    )
}

/// Pool a `[N, C, spatial...]` coeus CPU tensor into caller-owned output.
///
/// The input rank selects the monomorphized leto kernel: rank 3 pools over one
/// spatial axis, rank 4 over two, rank 5 over three.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::pooling_forward_into;
/// use leto_ops::PoolingMode;
///
/// let input_layout = Layout::new([1, 1, 3, 3].into());
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let output_layout = Layout::new([1, 1, 2, 2].into());
/// let mut output = [0.0_f64; 4];
/// pooling_forward_into(
///     PoolingMode::Maximum,
///     &input_layout,
///     &input,
///     2,
///     1,
///     0,
///     1,
///     &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [5.0, 6.0, 8.0, 9.0]);
///
/// pooling_forward_into(
///     PoolingMode::Average,
///     &input_layout,
///     &input,
///     2,
///     1,
///     0,
///     1,
///     &output_layout,
///     &mut output,
/// )
/// .unwrap();
/// assert_eq!(output, [3.0, 4.0, 6.0, 7.0]);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus PoolOps forward signature one-to-one"
)]
pub fn pooling_forward_into<T: LetoScalar>(
    mode: PoolingMode,
    input_layout: &CoeusLayout,
    input: &[T],
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    match input_layout.ndim() {
        3 => pooling_forward_n::<T, 3, 1>(
            mode,
            input_layout,
            input,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        4 => pooling_forward_n::<T, 4, 2>(
            mode,
            input_layout,
            input,
            kernel,
            stride,
            padding,
            dilation,
            output_layout,
            output,
        ),
        5 => pooling_forward_n::<T, 5, 3>(
            mode,
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
            reason: format!("pooling expects rank 3..=5 ([N, C, spatial...]), got {rank}"),
        }),
    }
}

/// Accumulate a pooling gradient into caller-owned `grad_input`.
///
/// The maximum path requires the forward `input`; the average path passes
/// `None` and derives its geometry from `grad_input`. Accumulation (`+=`)
/// matches the historical coeus CPU backward semantics.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::pooling_backward_into;
/// use leto_ops::PoolingMode;
///
/// let grad_output_layout = Layout::new([1, 1, 2, 2].into());
/// let grad_output = [1.0_f64; 4];
/// let grad_input_layout = Layout::new([1, 1, 3, 3].into());
/// let mut grad_input = [0.0_f64; 9];
/// pooling_backward_into(
///     PoolingMode::Average,
///     &grad_output_layout,
///     &grad_output,
///     None,
///     2,
///     1,
///     0,
///     1,
///     &grad_input_layout,
///     &mut grad_input,
/// )
/// .unwrap();
/// assert_eq!(
///     grad_input,
///     [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
/// );
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "dispatch mirrors the coeus PoolOps backward signature one-to-one"
)]
pub fn pooling_backward_into<T: LetoScalar>(
    mode: PoolingMode,
    grad_output_layout: &CoeusLayout,
    grad_output: &[T],
    input: Option<(&CoeusLayout, &[T])>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input_layout: &CoeusLayout,
    grad_input: &mut [T],
) -> Result<()> {
    match grad_output_layout.ndim() {
        3 => pooling_backward_n::<T, 3, 1>(
            mode,
            grad_output_layout,
            grad_output,
            input,
            kernel,
            stride,
            padding,
            dilation,
            grad_input_layout,
            grad_input,
        ),
        4 => pooling_backward_n::<T, 4, 2>(
            mode,
            grad_output_layout,
            grad_output,
            input,
            kernel,
            stride,
            padding,
            dilation,
            grad_input_layout,
            grad_input,
        ),
        5 => pooling_backward_n::<T, 5, 3>(
            mode,
            grad_output_layout,
            grad_output,
            input,
            kernel,
            stride,
            padding,
            dilation,
            grad_input_layout,
            grad_input,
        ),
        rank => Err(LetoError::StorageError {
            reason: format!("pooling expects rank 3..=5 ([N, C, spatial...]), got {rank}"),
        }),
    }
}
