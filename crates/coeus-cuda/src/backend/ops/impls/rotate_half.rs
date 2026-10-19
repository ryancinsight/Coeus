use crate::{backend::CudaBackend, CudaBackendError};
use coeus_accelerator::{rotate_half, RotateHalfProvider};
use coeus_core::Layout;

impl RotateHalfProvider<f32> for CudaBackend {
    type Operations = hephaestus_cuda::CudaElementwiseOps;
}

impl coeus_ops::RotateHalfOps<f32> for CudaBackend {
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<f32>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        rotate_half::<Self, _>(input.buffer(), layout)
            .map(|buffer| {
                // SAFETY: rotate_half initializes every output element on success.
                unsafe { coeus_accelerator::HephaestusStorage::from_buffer(buffer) }
            })
            .map_err(|source| CudaBackendError::dispatch("rotate_half", source))
    }
}
