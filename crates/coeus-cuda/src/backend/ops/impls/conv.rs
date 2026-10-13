use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Float;
use coeus_hephaestus::{
    convolution_backward, convolution_forward, convolution_transposed_backward,
    convolution_transposed_forward, ConvolutionBackend,
};
use coeus_ops::{ConvolutionBackward, ConvolutionForward};
use hephaestus_core::ConvolutionOps;
use hephaestus_cuda::{CudaConvolutionOps, CudaDevice};

impl<T> ConvolutionBackend<T> for CudaBackend
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
{
    type Operations = CudaConvolutionOps;
}

impl<T> coeus_ops::ConvOps<T> for CudaBackend
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
{
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        convolution_forward::<Self, T, R, D>(request.into(), stride, padding, dilation)
    }

    fn convolution_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        convolution_backward::<Self, T, R, D>(request.into(), stride, padding, dilation)
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
        convolution_transposed_forward::<Self, T, R, D>(
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
        convolution_transposed_backward::<Self, T, R, D>(
            request.into(),
            stride,
            padding,
            output_padding,
            dilation,
        )
    }
}
