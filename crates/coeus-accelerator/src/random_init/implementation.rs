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
            .map(|buffer| {
                // SAFETY: uniform initialization writes every element.
                unsafe { HephaestusStorage::from_buffer(buffer) }
            })
            .map_err(|source| {
                P::Error::from(HephaestusBackendError::device(
                    "uniform initialization",
                    source,
                ))
            })
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        normal::<P, T>(layout, mean, std_dev, seed)
            .map(|buffer| {
                // SAFETY: normal initialization writes every element.
                unsafe { HephaestusStorage::from_buffer(buffer) }
            })
            .map_err(|source| {
                P::Error::from(HephaestusBackendError::device(
                    "normal initialization",
                    source,
                ))
            })
    }
}
