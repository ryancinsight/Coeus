use crate::backend::{get_cuda_device, CudaBackend};
use coeus_core::{BackendError, Scalar};
use coeus_hephaestus::HephaestusSeamBackend;
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_cuda::CudaDevice;

impl HephaestusSeamBackend for CudaBackend {
    type Device = CudaDevice;

    fn device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn configuration_error(operation: &'static str, reason: String) -> Self::Error {
        crate::CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        crate::CudaBackendError::dispatch(operation, source)
    }
}
