use super::super::{CpuBackend, convolution};
use crate::backend_ops::traits::{ConvOps, ConvolutionBackward, ConvolutionForward};
use coeus_core::{CpuAddressableStorageMut, Float, Scalar};

impl<T, B> ConvOps<T> for B
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        convolution::regular_forward::<B, T, R, D>(
            convolution::Forward {
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
        convolution::regular_backward::<B, T, R, D>(
            convolution::Backward {
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
        convolution::transposed_forward::<B, T, R, D>(
            convolution::Forward {
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
        convolution::transposed_backward::<B, T, R, D>(
            convolution::Backward {
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
