//! Transposed convolution routing.

use leto::{Result, TransposedConvolutionParameters};
use leto_ops::{Scalar, TransposedConvolutionGradients};

use super::{
    views::{BackwardViews, ForwardViews},
    ConvolutionBackward, ConvolutionForward,
};

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
    let ForwardViews {
        input,
        weight,
        bias,
        mut output,
    } = super::views::forward::<T, R>(operands)?;

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
    let BackwardViews {
        input,
        weight,
        grad_output,
        mut grad_input,
        mut grad_weight,
        mut grad_bias,
    } = super::views::backward::<T, R>(operands)?;
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
