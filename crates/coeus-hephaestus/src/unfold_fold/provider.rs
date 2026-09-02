use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar};
use hephaestus_core::{ComputeDevice, HephaestusError, SlidingWindowOps};

/// Provider-owned sliding-window operation marker.
pub trait UnfoldFoldProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus sliding-window operations selected by this provider.
    type Operations: SlidingWindowOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus sliding-window provider.
pub trait UnfoldFoldBackend<T>: ComputeBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized sliding-window operations for the selected device.
    type Operations: SlidingWindowOps<Self::Device, T> + Default;

    /// Return the lazily acquired provider device.
    fn unfold_fold_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn unfold_fold_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn unfold_fold_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn unfold_fold_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P, T> UnfoldFoldBackend<T> for HephaestusBackend<P>
where
    P: UnfoldFoldProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Device = P::Device;
    type Operations = P::Operations;

    fn unfold_fold_device() -> &'static Self::Device {
        P::device()
    }

    fn unfold_fold_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn unfold_fold_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        HephaestusBackendError::device(
            operation,
            HephaestusError::InvalidConfiguration { message: reason },
        )
    }

    fn unfold_fold_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }
}
