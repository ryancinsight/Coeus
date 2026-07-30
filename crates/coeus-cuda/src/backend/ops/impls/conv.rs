use crate::backend::{CudaBackend, CudaScalar, get_cuda_device};
use coeus_core::{BackendError, Float};
use coeus_hephaestus::{
    ConvolutionBackend, convolution_backward, convolution_forward, convolution_transposed_backward,
    convolution_transposed_forward,
};
use coeus_ops::{ConvolutionBackward, ConvolutionForward};
use hephaestus_core::{ComputeDevice, ConvolutionOps, HephaestusError};
use hephaestus_cuda::{CudaConvolutionOps, CudaDevice};

impl<T> ConvolutionBackend<T> for CudaBackend
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
{
    type Device = CudaDevice;
    type Operations = CudaConvolutionOps;

    fn convolution_device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn convolution_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn convolution_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        crate::CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn convolution_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        crate::CudaBackendError::dispatch(operation, source)
    }
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
