use hephaestus_core::HephaestusError;
use hephaestus_cuda::CudaDevice;

pub(crate) fn device_available() -> bool {
    match CudaDevice::try_default() {
        Ok(device) => {
            // Release the probe before the test acquires its operation context;
            // retaining it would hide final-owner context lifecycle failures.
            drop(device);
            true
        }
        Err(HephaestusError::AdapterUnavailable { .. })
            if std::env::var("HEPHAESTUS_CUDA_REQUIRE_DEVICE").as_deref() != Ok("1") =>
        {
            false
        }
        Err(error) => panic!("CUDA test device acquisition failed: {error}"),
    }
}
