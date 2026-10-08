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

/// True when a device is present and the operation device serves f64 shaders.
///
/// f64 dispatch needs `SHADER_F64`, requested optionally at device creation;
/// tests that need it gate on this probe and skip where the adapter lacks it.
pub(crate) fn device_supports_f64(label: &str) -> bool {
    device_available(label) && coeus_wgpu::WgpuBackend::supports_f64()
}

/// True when a device is present and the operation device serves f16 shaders.
///
/// f16 dispatch needs `SHADER_F16`, requested optionally at device creation;
/// tests that need it gate on this probe and skip where the adapter lacks it.
pub(crate) fn device_supports_f16(label: &str) -> bool {
    device_available(label) && coeus_wgpu::WgpuBackend::supports_f16()
}
