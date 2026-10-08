use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::ComputeBackend;
use hephaestus_core::{ComputeDevice, FixedFd3DOps, HephaestusError};

/// Provider-owned fixed-scheme sweep marker.
///
/// A provider implements this when its device has the fixed-scheme kernels.
/// There is no scalar parameter: the provider states the sweeps in `f32`,
/// because WGSL does not guarantee `f64` storage and a generic scalar at
/// this boundary would be falsely generic.
pub trait FixedFdProvider: HephaestusProvider {
    /// Monomorphized Hephaestus sweep operations selected by this provider.
    type Operations: FixedFd3DOps<Self::Device> + Default;

    /// Borrow the compiled sweep kernel, compiling it on first use.
    ///
    /// The sweep seam carries no preparation step — scheme and spacing arrive
    /// per call — so each provider caches the one compiled kernel behind a
    /// static rather than recompiling pipelines per sweep. A failed
    /// compilation is not cached, so the next sweep retries.
    fn fixed_fd_kernel(
    ) -> hephaestus_core::Result<&'static <Self::Operations as FixedFd3DOps<Self::Device>>::FixedFd3D>;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus sweep provider.
pub trait FixedFdBackend: ComputeBackend {
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Monomorphized sweep operations for the selected device.
    type Operations: FixedFd3DOps<Self::Device> + Default;
    /// Compiled sweep kernel, cached one per backend.
    type Kernel: Send + Sync + 'static;

    /// Return the lazily acquired provider device.
    fn fixed_fd_device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn fixed_fd_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32>;

    /// Borrow the provider's cached sweep kernel.
    fn fixed_fd_kernel() -> Result<&'static Self::Kernel, Self::Error>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn fixed_fd_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn fixed_fd_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P> FixedFdBackend for HephaestusBackend<P>
where
    P: FixedFdProvider,
    <P::Operations as FixedFd3DOps<P::Device>>::FixedFd3D: Send + Sync,
{
    type Device = P::Device;
    type Operations = P::Operations;
    type Kernel = <P::Operations as FixedFd3DOps<P::Device>>::FixedFd3D;

    fn fixed_fd_device() -> &'static Self::Device {
        P::device()
    }

    fn fixed_fd_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn fixed_fd_kernel() -> Result<&'static Self::Kernel, Self::Error> {
        P::fixed_fd_kernel()
            .map_err(|source| Self::fixed_fd_dispatch_error("fixed_fd_kernel", source))
    }

    fn fixed_fd_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        crate::error::invalid_configuration_error(operation, reason)
    }

    fn fixed_fd_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }
}
