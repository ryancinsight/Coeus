//! Regular convolution routing.

use leto::{ConvolutionParameters, Result};
use leto_ops::Scalar;

use super::{
    views::{BackwardViews, ForwardViews},
    ConvolutionBackward, ConvolutionForward,
};

/// Execute regular convolution directly through Leto over borrowed Coeus
/// storage.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when parameters, shapes, layouts, or storage
/// violate the provider contract.
pub fn convolution_forward_into<T: Scalar, const R: usize, const D: usize>(
    operands: ConvolutionForward<'_, T>,
    parameters: ConvolutionParameters<D>,
) -> Result<()> {
    let ForwardViews {
        input,
        weight,
        bias,
        mut output,
    } = super::views::forward::<T, R>(operands)?;

    leto_ops::convolution_forward_into(&input, &weight, bias.as_ref(), parameters, &mut output)
}

/// Accumulate regular-convolution gradients directly through Leto over
/// borrowed Coeus storage.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when parameters, shapes, layouts, selected
/// gradients, or storage violate the provider contract.
pub fn convolution_backward_accumulate<T: Scalar, const R: usize, const D: usize>(
    operands: ConvolutionBackward<'_, T>,
    parameters: ConvolutionParameters<D>,
) -> Result<()> {
    let BackwardViews {
        input,
        weight,
        grad_output,
        mut grad_input,
        mut grad_weight,
        mut grad_bias,
    } = super::views::backward::<T, R>(operands)?;

    leto_ops::convolution_backward_accumulate(
        &input,
        &weight,
        &grad_output,
        parameters,
        grad_input.as_mut(),
        grad_weight.as_mut(),
        grad_bias.as_mut(),
    )
}
