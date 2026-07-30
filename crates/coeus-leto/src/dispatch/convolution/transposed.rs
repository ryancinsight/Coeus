//! Transposed convolution routing.

use leto::{ArrayView, ArrayViewMut, Result, TransposedConvolutionParameters};
use leto_ops::{Scalar, TransposedConvolutionGradients};

use super::{bias_layout, ConvolutionBackward, ConvolutionForward};
use crate::{to_leto_view, to_leto_view_mut};

/// Execute transposed convolution directly through Leto over borrowed Coeus
/// storage.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when parameters, shapes, layouts, or storage
/// violate the provider contract.
pub fn convolution_transposed_forward_into<T: Scalar, const R: usize, const D: usize>(
    operands: ConvolutionForward<'_, T>,
    parameters: TransposedConvolutionParameters<D>,
) -> Result<()> {
    let input = to_leto_view::<T, R>(operands.input.layout, operands.input.data)?;
    let weight = to_leto_view::<T, R>(operands.weight.layout, operands.weight.data)?;
    let bias = operands
        .bias
        .map(|data| ArrayView::try_new(bias_layout(data.len())?, data))
        .transpose()?;
    let mut output = to_leto_view_mut::<T, R>(operands.output.layout, operands.output.data)?;

    leto_ops::convolution_transposed_forward_into(
        &input,
        &weight,
        bias.as_ref(),
        parameters,
        &mut output,
    )
}

/// Accumulate transposed-convolution gradients directly through Leto over
/// borrowed Coeus storage.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when parameters, shapes, layouts, selected
/// gradients, or storage violate the provider contract.
pub fn convolution_transposed_backward_accumulate<T: Scalar, const R: usize, const D: usize>(
    operands: ConvolutionBackward<'_, T>,
    parameters: TransposedConvolutionParameters<D>,
) -> Result<()> {
    let input = to_leto_view::<T, R>(operands.input.layout, operands.input.data)?;
    let weight = to_leto_view::<T, R>(operands.weight.layout, operands.weight.data)?;
    let grad_output = to_leto_view::<T, R>(operands.grad_output.layout, operands.grad_output.data)?;
    let mut grad_input = operands
        .gradients
        .input
        .map(|target| to_leto_view_mut::<T, R>(target.layout, target.data))
        .transpose()?;
    let mut grad_weight = operands
        .gradients
        .weight
        .map(|target| to_leto_view_mut::<T, R>(target.layout, target.data))
        .transpose()?;
    let mut grad_bias = operands
        .gradients
        .bias
        .map(|data| ArrayViewMut::try_new(bias_layout(data.len())?, data))
        .transpose()?;
    let gradients = TransposedConvolutionGradients::new(
        grad_input.as_mut(),
        grad_weight.as_mut(),
        grad_bias.as_mut(),
    );

    leto_ops::convolution_transposed_backward_accumulate(
        &input,
        &weight,
        &grad_output,
        parameters,
        gradients,
    )
}
