use crate::backend::{WgpuBackend, WgpuBackendError};
use coeus_core::{Layout, Scalar};
use coeus_hephaestus::{random_normal, random_uniform, RandomInitProvider};
use hephaestus_core::RandomInitOps;
use hephaestus_wgpu::{DialectScalar, WgpuDevice, WgpuRandomOps, Wgsl};

// NOTE: `WgpuScalar` is deliberately not required: it gates the fused
// dispatch, while random initialization is host-delegated through Leto,
// whose `DialectScalar<Wgsl>` admits f64 as well.
impl<T> RandomInitProvider<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuRandomOps: RandomInitOps<WgpuDevice, T>,
{
    type Operations = WgpuRandomOps;
}

impl<T> coeus_ops::RandomInitOps<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuRandomOps: RandomInitOps<WgpuDevice, T>,
{
    fn uniform_random(
        &self,
        layout: &Layout,
        low: T,
        high: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        random_uniform::<Self, _>(layout, low, high, seed)
            .map(coeus_hephaestus::HephaestusStorage::from_buffer)
            .map_err(|source| WgpuBackendError::dispatch("uniform initialization", source))
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        random_normal::<Self, _>(layout, mean, std_dev, seed)
            .map(coeus_hephaestus::HephaestusStorage::from_buffer)
            .map_err(|source| WgpuBackendError::dispatch("normal initialization", source))
    }
}
