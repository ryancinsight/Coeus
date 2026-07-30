use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar};
use hephaestus_core::{ComputeDevice, ConvolutionOps, HephaestusError};

/// Provider-owned scalar convolution operation marker.
pub trait ConvolutionProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operation marker selected by this provider.
    type Operations: ConvolutionOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus convolution
/// provider.
///
/// Implementations expose the selected device buffer without host transfer and
/// preserve backend-specific error types at the consumer boundary.
pub trait ConvolutionBackend<T>: ComputeBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized convolution operations for the selected device.
    type Operations: ConvolutionOps<Self::Device, T> + Default;

    /// Return the lazily acquired provider device.
    fn convolution_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn convolution_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn convolution_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn convolution_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P, T> ConvolutionBackend<T> for HephaestusBackend<P>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Device = P::Device;
    type Operations = P::Operations;

    fn convolution_device() -> &'static Self::Device {
        P::device()
    }

    fn convolution_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn convolution_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        HephaestusBackendError::device(
            operation,
            HephaestusError::InvalidConfiguration { message: reason },
        )
    }

    fn convolution_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }
}
