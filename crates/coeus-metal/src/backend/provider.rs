use coeus_hephaestus::{HephaestusBackendError, HephaestusProvider, SharedHephaestusError};
use hephaestus_metal::MetalDevice;
use std::sync::OnceLock;

/// Provider marker for the native Metal device.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalProvider;

// SAFETY: Metal buffers retain the WGPU/Metal device context and WGPU queue
// submission supplies the synchronization boundary required by the handle.
unsafe impl HephaestusProvider for MetalProvider {
    type Device = MetalDevice;
    const NAME: &'static str = "metal";

    fn try_device() -> Result<&'static Self::Device, HephaestusBackendError> {
        static DEVICE: OnceLock<Result<MetalDevice, SharedHephaestusError>> = OnceLock::new();
        DEVICE
            .get_or_init(|| MetalDevice::try_default().map_err(SharedHephaestusError::new))
            .as_ref()
            .map_err(|source| HephaestusBackendError::initialization(Self::NAME, source.clone()))
    }
}
