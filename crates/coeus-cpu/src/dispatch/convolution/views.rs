use leto::{ArrayView, ArrayViewMut, Result};
use leto_ops::Scalar;

use super::{bias_layout, ConvolutionBackward, ConvolutionForward};
use crate::{to_leto_view, to_leto_view_mut};

pub(super) struct ForwardViews<'a, T: Scalar, const R: usize> {
    pub input: ArrayView<'a, T, R>,
    pub weight: ArrayView<'a, T, R>,
    pub bias: Option<ArrayView<'a, T, 1>>,
    pub output: ArrayViewMut<'a, T, R>,
}

pub(super) struct BackwardViews<'a, T: Scalar, const R: usize> {
    pub input: ArrayView<'a, T, R>,
    pub weight: ArrayView<'a, T, R>,
    pub grad_output: ArrayView<'a, T, R>,
    pub grad_input: Option<ArrayViewMut<'a, T, R>>,
    pub grad_weight: Option<ArrayViewMut<'a, T, R>>,
    pub grad_bias: Option<ArrayViewMut<'a, T, 1>>,
}

pub(super) fn forward<T: Scalar, const R: usize>(
    operands: ConvolutionForward<'_, T>,
) -> Result<ForwardViews<'_, T, R>> {
    let input = to_leto_view::<T, R>(operands.input.layout, operands.input.data)?;
    let weight = to_leto_view::<T, R>(operands.weight.layout, operands.weight.data)?;
    let bias = operands
        .bias
        .map(|data| ArrayView::try_new(bias_layout(data.len())?, data))
        .transpose()?;
    let output = to_leto_view_mut::<T, R>(operands.output.layout, operands.output.data)?;

    Ok(ForwardViews {
        input,
        weight,
        bias,
        output,
    })
}

pub(super) fn backward<T: Scalar, const R: usize>(
    operands: ConvolutionBackward<'_, T>,
) -> Result<BackwardViews<'_, T, R>> {
    let input = to_leto_view::<T, R>(operands.input.layout, operands.input.data)?;
    let weight = to_leto_view::<T, R>(operands.weight.layout, operands.weight.data)?;
    let grad_output = to_leto_view::<T, R>(operands.grad_output.layout, operands.grad_output.data)?;
    let grad_input = operands
        .gradients
        .input
        .map(|target| to_leto_view_mut::<T, R>(target.layout, target.data))
        .transpose()?;
    let grad_weight = operands
        .gradients
        .weight
        .map(|target| to_leto_view_mut::<T, R>(target.layout, target.data))
        .transpose()?;
    let grad_bias = operands
        .gradients
        .bias
        .map(|data| ArrayViewMut::try_new(bias_layout(data.len())?, data))
        .transpose()?;

    Ok(BackwardViews {
        input,
        weight,
        grad_output,
        grad_input,
        grad_weight,
        grad_bias,
    })
}
