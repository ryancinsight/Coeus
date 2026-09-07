use hephaestus_core::HephaestusError;
use hephaestus_wgpu::WgpuDevice;

pub(crate) fn device_available(label: &str) -> bool {
    match WgpuDevice::try_default(label) {
        Ok(device) => {
            // Release the probe before the test acquires its operation device;
            // retaining it would change the device lifecycle being exercised.
            drop(device);
            true
        }
        Err(HephaestusError::AdapterUnavailable { .. })
            if std::env::var("HEPHAESTUS_WGPU_REQUIRE_DEVICE").as_deref() != Ok("1") =>
        {
            false
        }
        Err(error) => panic!("WGPU test device acquisition failed for {label}: {error:?}"),
    }
}
