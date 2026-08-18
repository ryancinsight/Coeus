use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar};
use hephaestus_core::{ComputeDevice, DenseProductOps, HephaestusError};

/// Provider-owned scalar dense-product operation marker.
pub trait MatmulProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operation marker selected by this provider.
    type Operations: DenseProductOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus dense-product
/// provider.
///
/// Implementations expose the selected device buffer without host transfer and
/// preserve backend-specific error types at the consumer boundary. A backend
/// that is not [`HephaestusBackend`] implements this trait directly and reuses
/// [`crate::matmul`] rather than carrying its own kernel.
pub trait MatmulBackend<T>: ComputeBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized dense-product operations for the selected device.
    type Operations: DenseProductOps<Self::Device, T> + Default;

    /// Return the lazily acquired provider device.
    fn matmul_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn matmul_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T>;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn matmul_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P, T> MatmulBackend<T> for HephaestusBackend<P>
where
    P: MatmulProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Device = P::Device;
    type Operations = P::Operations;

    fn matmul_device() -> &'static Self::Device {
        P::device()
    }

    fn matmul_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn matmul_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }
}
