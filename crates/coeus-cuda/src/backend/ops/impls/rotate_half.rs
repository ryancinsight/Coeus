use crate::{backend::CudaBackend, CudaBackendError, CudaStorage};
use coeus_core::Layout;
use coeus_hephaestus::{rotate_half, RotateHalfProvider};

impl RotateHalfProvider<f32> for CudaBackend {
    type Operations = hephaestus_cuda::CudaElementwiseOps;
}

impl coeus_ops::RotateHalfOps<f32> for CudaBackend {
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<f32>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        rotate_half::<Self, _>(input.buffer.as_ref(), layout)
            .map(CudaStorage::from_buffer)
            .map_err(|source| CudaBackendError::dispatch("rotate_half", source))
    }
}
