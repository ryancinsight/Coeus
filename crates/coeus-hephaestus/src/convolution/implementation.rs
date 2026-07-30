use super::{dispatch, provider::ConvolutionProvider};
use crate::HephaestusBackend;
use coeus_core::{Float, Scalar};
use coeus_ops::{ConvOps, ConvolutionBackward, ConvolutionForward};

impl<P, T> ConvOps<T> for HephaestusBackend<P>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        dispatch::regular_forward::<P, T, R, D>(
            dispatch::Forward {
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                bias: request.bias,
                output: request.output,
                output_layout: request.output_layout,
            },
            stride,
            padding,
            dilation,
        )
    }

    fn convolution_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        dispatch::regular_backward::<P, T, R, D>(
            dispatch::Backward {
                grad_output: request.grad_output,
                grad_output_layout: request.grad_output_layout,
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                grad_input: request.grad_input,
                grad_input_layout: request.grad_input_layout,
                grad_weight: request.grad_weight,
                grad_weight_layout: request.grad_weight_layout,
                grad_bias: request.grad_bias,
            },
            stride,
            padding,
            dilation,
        )
    }

    fn convolution_transposed_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        dispatch::transposed_forward::<P, T, R, D>(
            dispatch::Forward {
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                bias: request.bias,
                output: request.output,
                output_layout: request.output_layout,
            },
            stride,
            padding,
            output_padding,
            dilation,
        )
    }

    fn convolution_transposed_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        dispatch::transposed_backward::<P, T, R, D>(
            dispatch::Backward {
                grad_output: request.grad_output,
                grad_output_layout: request.grad_output_layout,
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                grad_input: request.grad_input,
                grad_input_layout: request.grad_input_layout,
                grad_weight: request.grad_weight,
                grad_weight_layout: request.grad_weight_layout,
                grad_bias: request.grad_bias,
            },
            stride,
            padding,
            output_padding,
            dilation,
        )
    }
}
