use coeus_core::{BackendError, CpuAddressableStorageMut, Layout, Scalar};

use super::CpuBackend;
use crate::RandomInitOps;

fn provider_error(operation: &'static str, source: leto::LetoError) -> BackendError {
    BackendError::Storage {
        operation,
        reason: source.to_string(),
    }
}

impl<T, B> RandomInitOps<T> for B
where
    T: Scalar + coeus_leto::RandomScalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    fn uniform_random(
        &self,
        layout: &Layout,
        low: T,
        high: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // Leto's destination contract requires initialized `T` values before a
        // mutable slice can exist. Keep the single final allocation while
        // avoiding a typed reference over uninitialized storage.
        let mut storage = self.allocate_zeroed(layout.numel());
        coeus_leto::uniform_values_into(layout, storage.as_mut_slice(), low, high, seed)
            .map_err(|source| provider_error("uniform initialization", source))?;
        Ok(storage)
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // See `uniform_random`: zero-initialization is the validity boundary
        // for the destination-writing provider API.
        let mut storage = self.allocate_zeroed(layout.numel());
        coeus_leto::normal_values_into(layout, storage.as_mut_slice(), mean, std_dev, seed)
            .map_err(|source| provider_error("normal initialization", source))?;
        Ok(storage)
    }
}
