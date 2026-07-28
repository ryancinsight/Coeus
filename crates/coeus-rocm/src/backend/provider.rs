use coeus_hephaestus::{HephaestusBackendError, HephaestusProvider, SharedHephaestusError};
use hephaestus_rocm::RocmDevice;
use std::sync::OnceLock;

/// Provider marker for the native ROCm device.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmProvider;

// SAFETY: ROCm buffers retain their owning context and HIP launches bind that
// context before accessing the allocation; the handle is thread-transferable.
unsafe impl HephaestusProvider for RocmProvider {
    type Device = RocmDevice;
    const NAME: &'static str = "rocm";

    fn try_device() -> Result<&'static Self::Device, HephaestusBackendError> {
        static DEVICE: OnceLock<Result<RocmDevice, SharedHephaestusError>> = OnceLock::new();
        DEVICE
            .get_or_init(|| RocmDevice::try_default().map_err(SharedHephaestusError::new))
            .as_ref()
            .map_err(|source| HephaestusBackendError::initialization(Self::NAME, source.clone()))
    }
}
