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
        dispatch::regular_forward::<HephaestusBackend<P>, T, R, D>(
            request.into(),
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
        dispatch::regular_backward::<HephaestusBackend<P>, T, R, D>(
            request.into(),
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
        dispatch::transposed_forward::<HephaestusBackend<P>, T, R, D>(
            request.into(),
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
        dispatch::transposed_backward::<HephaestusBackend<P>, T, R, D>(
            request.into(),
            stride,
            padding,
            output_padding,
            dilation,
        )
    }
}
