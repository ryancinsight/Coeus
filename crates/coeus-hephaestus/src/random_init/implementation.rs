use super::{normal, uniform, RandomInitProvider};
use crate::{HephaestusBackend, HephaestusBackendError, HephaestusStorage};
use coeus_core::{Layout, Scalar};

impl<P, T> coeus_ops::RandomInitOps<T> for HephaestusBackend<P>
where
    P: RandomInitProvider<T>,
    T: Scalar,
{
    fn uniform_random(
        &self,
        layout: &Layout,
        low: T,
        high: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        uniform::<P, T>(layout, low, high, seed)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("uniform initialization", source))
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        normal::<P, T>(layout, mean, std_dev, seed)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("normal initialization", source))
    }
}
