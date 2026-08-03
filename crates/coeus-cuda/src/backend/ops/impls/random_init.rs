use crate::backend::CudaBackend;
use crate::{CudaBackendError, CudaStorage};
use coeus_core::Layout;
use coeus_hephaestus::{random_normal, random_uniform, RandomInitProvider};

impl RandomInitProvider<f32> for CudaBackend {
    type Operations = hephaestus_cuda::CudaRandomOps;
}

impl coeus_ops::RandomInitOps<f32> for CudaBackend {
    fn uniform_random(
        &self,
        layout: &Layout,
        low: f32,
        high: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_uniform::<Self, _>(layout, low, high, seed)
            .map(CudaStorage::from_buffer)
            .map_err(|source| CudaBackendError::dispatch("uniform initialization", source))
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: f32,
        std_dev: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_normal::<Self, _>(layout, mean, std_dev, seed)
            .map(CudaStorage::from_buffer)
            .map_err(|source| CudaBackendError::dispatch("normal initialization", source))
    }
}
