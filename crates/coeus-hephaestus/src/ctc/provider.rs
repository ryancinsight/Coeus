use coeus_core::{BackendError, ComputeBackend};
use hephaestus_core::{ComputeDevice, CtcOps, HephaestusError};

use crate::{
    error::invalid_configuration_error, HephaestusBackend, HephaestusBackendError,
    HephaestusProvider,
};

/// A provider with compiled CTC step kernels and a per-provider kernel cache.
pub trait CtcProvider: HephaestusProvider {
    /// Step kernels bound to this provider's device.
    type Operations: CtcOps<Self::Device> + Default;

    /// Borrow the cached step kernels for this provider.
    ///
    /// # Errors
    ///
    /// Returns the provider's kernel compilation failure.
    fn ctc_kernel(
    ) -> Result<&'static <Self::Operations as CtcOps<Self::Device>>::Ctc, HephaestusError>;
}

/// The CTC contract a Coeus backend serves through a Hephaestus provider.
pub trait CtcBackend: ComputeBackend {
    /// Provider device behind this backend.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Step kernels for the provider device.
    type Operations: CtcOps<Self::Device> + Default;
    /// Cached kernels shared across dispatches on this backend.
    type Kernel: Send + Sync + 'static;

    /// Borrow the process-global provider device.
    fn ctc_device() -> &'static Self::Device;

    /// Borrow the raw provider buffer behind Coeus device storage.
    fn ctc_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32>;

    /// Borrow the cached step kernels for this backend.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation failure.
    fn ctc_kernel() -> Result<&'static Self::Kernel, Self::Error>;

    /// Map a malformed CTC configuration to the backend's error.
    fn ctc_configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a provider CTC failure to the backend's error.
    fn ctc_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;

    /// Map an impossible alignment to the backend's error.
    ///
    /// Mirrors the CPU implementation, which rejects the backward pass
    /// before touching the gradient when a sample has no valid alignment.
    fn ctc_impossible_error(operation: &'static str, sample: usize) -> Self::Error;

    /// Map non-finite CTC arithmetic to the backend's error.
    fn ctc_arithmetic_error(operation: &'static str, sample: usize) -> Self::Error;
}

impl<P> CtcBackend for HephaestusBackend<P>
where
    P: CtcProvider,
    <P::Operations as CtcOps<P::Device>>::Ctc: Send + Sync,
{
    type Device = P::Device;
    type Operations = P::Operations;
    type Kernel = <P::Operations as CtcOps<P::Device>>::Ctc;

    fn ctc_device() -> &'static Self::Device {
        P::device()
    }

    fn ctc_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn ctc_kernel() -> Result<&'static Self::Kernel, Self::Error> {
        P::ctc_kernel().map_err(|source| HephaestusBackendError::device("ctc", source))
    }

    fn ctc_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        invalid_configuration_error(operation, reason)
    }

    fn ctc_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::device(operation, source)
    }

    fn ctc_impossible_error(operation: &'static str, sample: usize) -> Self::Error {
        BackendError::UndefinedGradient { operation, sample }.into()
    }

    fn ctc_arithmetic_error(operation: &'static str, sample: usize) -> Self::Error {
        BackendError::NonFiniteSample { operation, sample }.into()
    }
}
