use crate::backend::{try_get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_accelerator::HephaestusSeamBackend;
use coeus_core::{BackendError, Scalar};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::WgpuDevice;

impl HephaestusSeamBackend for WgpuBackend {
    type Device = WgpuDevice;

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        try_get_wgpu_context().map(|context| &context.hephaestus_device)
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
