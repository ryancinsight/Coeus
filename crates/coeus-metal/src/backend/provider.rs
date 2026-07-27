use coeus_hephaestus::HephaestusProvider;
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

    fn device() -> &'static Self::Device {
        static DEVICE: OnceLock<MetalDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MetalDevice::try_default().expect("Metal device acquisition failed"))
    }
}
