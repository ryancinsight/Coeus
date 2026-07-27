use coeus_hephaestus::HephaestusProvider;
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

    fn device() -> &'static Self::Device {
        static DEVICE: OnceLock<RocmDevice> = OnceLock::new();
        DEVICE.get_or_init(|| RocmDevice::try_default().expect("ROCm device acquisition failed"))
    }
}
