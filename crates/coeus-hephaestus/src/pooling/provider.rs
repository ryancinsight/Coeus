use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar};
use hephaestus_core::{ComputeDevice, HephaestusError, PoolingOps};

/// Provider-owned pooling operation marker.
///
/// # Safety
///
/// Successful forward operations selected by `Operations` must initialize
/// every logical output element without reading prior contents. Backward
/// operations accumulate and may read their gradient destinations, which
/// must be initialized before dispatch. A dispatch error may leave an output
/// partially written.
pub unsafe trait PoolingProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus pooling operations selected by this provider.
    type Operations: PoolingOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus pooling provider.
pub trait PoolingBackend<T>: ComputeBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized pooling operations for the selected device.
    type Operations: PoolingOps<Self::Device, T> + Default;

    /// Return the lazily acquired provider device.
    fn pooling_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn pooling_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn pooling_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn pooling_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P, T> PoolingBackend<T> for HephaestusBackend<P>
where
    P: PoolingProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Device = P::Device;
    type Operations = P::Operations;

    fn pooling_device() -> &'static Self::Device {
        P::device()
    }

    fn pooling_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn pooling_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        P::Error::from(crate::error::invalid_configuration_error(operation, reason))
    }

    fn pooling_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        P::Error::from(HephaestusBackendError::device(operation, source))
    }
}
