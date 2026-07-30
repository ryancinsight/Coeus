use crate::backend::{WgpuBackend, WgpuBackendError, WgpuScalar, get_wgpu_context};
use coeus_core::{BackendError, Float};
use coeus_hephaestus::{
    ConvolutionBackend, convolution_backward, convolution_forward, convolution_transposed_backward,
    convolution_transposed_forward,
};
use coeus_ops::{ConvolutionBackward, ConvolutionForward};
use hephaestus_core::{ComputeDevice, ConvolutionOps, HephaestusError};
use hephaestus_wgpu::{DialectScalar, WgpuConvolutionOps, WgpuDevice, Wgsl};

impl<T> ConvolutionBackend<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    type Device = WgpuDevice;
    type Operations = WgpuConvolutionOps;

    fn convolution_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn convolution_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn convolution_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn convolution_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::ConvOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
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
