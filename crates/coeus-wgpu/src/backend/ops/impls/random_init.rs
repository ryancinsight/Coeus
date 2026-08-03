use crate::backend::{WgpuBackend, WgpuBackendError};
use crate::storage::WgpuStorage;
use coeus_core::Layout;
use coeus_hephaestus::{random_normal, random_uniform, RandomInitProvider};

impl RandomInitProvider<f32> for WgpuBackend {
    type Operations = hephaestus_wgpu::WgpuRandomOps;
}

impl coeus_ops::RandomInitOps<f32> for WgpuBackend {
    fn uniform_random(
        &self,
        layout: &Layout,
        low: f32,
        high: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_uniform::<Self, _>(layout, low, high, seed)
            .map(WgpuStorage::from_buffer)
            .map_err(|source| WgpuBackendError::dispatch("uniform initialization", source))
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: f32,
        std_dev: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_normal::<Self, _>(layout, mean, std_dev, seed)
            .map(WgpuStorage::from_buffer)
            .map_err(|source| WgpuBackendError::dispatch("normal initialization", source))
    }
}
