use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::ComputeBackend;
use hephaestus_core::{ComputeDevice, HephaestusError, Staggered3DOps};

/// Provider-owned staggered gradient/divergence marker.
///
/// A provider implements this when its device has the staggered kernels. There
/// is no scalar parameter: the provider states the pair in `f32`, because WGSL
/// does not guarantee `f64` storage and a generic scalar at this boundary would
/// be falsely generic.
pub trait StaggeredProvider: HephaestusProvider {
    /// Monomorphized Hephaestus staggered operations selected by this provider.
    type Operations: Staggered3DOps<Self::Device> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus staggered provider.
pub trait StaggeredBackend: ComputeBackend {
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized staggered operations for the selected device.
    type Operations: Staggered3DOps<Self::Device> + Default;

    /// Return the lazily acquired provider device.
    fn staggered_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn staggered_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn staggered_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn staggered_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P> StaggeredBackend for HephaestusBackend<P>
where
    P: StaggeredProvider,
{
    type Device = P::Device;
    type Operations = P::Operations;

    fn staggered_device() -> &'static Self::Device {
        P::device()
    }

    fn staggered_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn staggered_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        HephaestusBackendError::device(
            operation,
            HephaestusError::InvalidConfiguration { message: reason },
        )
    }

    fn staggered_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }
}
