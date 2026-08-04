use crate::{backend::WgpuBackend, storage::WgpuStorage};
use coeus_core::Layout;
use coeus_hephaestus::{rotate_half, RotateHalfProvider};

impl RotateHalfProvider<f32> for WgpuBackend {
    type Operations = hephaestus_wgpu::WgpuElementwiseOps;
}

impl coeus_ops::RotateHalfOps<f32> for WgpuBackend {
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<f32>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        rotate_half::<Self, _>(input.buffer.as_ref(), layout)
            .map(WgpuStorage::from_buffer)
            .map_err(|source| crate::backend::WgpuBackendError::dispatch("rotate_half", source))
    }
}
