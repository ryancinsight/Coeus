use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_core::{BackendError, Scalar};
use coeus_hephaestus::HephaestusSeamBackend;
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::WgpuDevice;

impl HephaestusSeamBackend for WgpuBackend {
    type Device = WgpuDevice;

    fn device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}
