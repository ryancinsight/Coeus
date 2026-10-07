use crate::backend::{CudaBackend, CudaScalar};
use crate::CudaBackendError;
use coeus_core::Layout;
use coeus_hephaestus::{random_normal, random_uniform, RandomInitProvider};
use hephaestus_core::RandomInitOps;
use hephaestus_cuda::{CudaDevice, CudaRandomOps};

impl<T> RandomInitProvider<T> for CudaBackend
where
    T: CudaScalar,
    CudaRandomOps: RandomInitOps<CudaDevice, T>,
{
    type Operations = CudaRandomOps;
}

impl<T> coeus_ops::RandomInitOps<T> for CudaBackend
where
    T: CudaScalar,
    CudaRandomOps: RandomInitOps<CudaDevice, T>,
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
            .map_err(|source| CudaBackendError::dispatch("uniform initialization", source))
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
            .map_err(|source| CudaBackendError::dispatch("normal initialization", source))
    }
}
