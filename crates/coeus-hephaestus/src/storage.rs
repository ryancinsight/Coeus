use crate::reduction::HephaestusProvider;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use std::{marker::PhantomData, sync::Arc};
use themis::{MemoryTier, PlacementHint};

/// Reference-counted Coeus storage backed by one Hephaestus device buffer.
pub struct HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod,
{
    pub(crate) buffer: Arc<<P::Device as ComputeDevice>::Buffer<T>>,
    marker: PhantomData<P>,
}

impl<P, T> Clone for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod,
{
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            marker: PhantomData,
        }
    }
}

impl<P, T> HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar + bytemuck::Pod,
{
    /// Allocate zeroed storage in the provider's device tier.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let buffer = P::device()
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Hephaestus provider allocation failed");
        Self {
            buffer: Arc::new(buffer),
            marker: PhantomData,
        }
    }

    /// Borrow the typed Hephaestus buffer for provider dispatch.
    #[must_use]
    pub fn buffer(&self) -> &<P::Device as ComputeDevice>::Buffer<T> {
        self.buffer.as_ref()
    }
}

impl<P, T> coeus_core::storage::private::Sealed for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod,
{
}

// SAFETY: `HephaestusProvider` requires its device buffers to be safe to move
// between threads while the provider owns the device synchronization contract.
unsafe impl<P, T> Send for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod + Send,
{
}

// SAFETY: `HephaestusProvider` requires shared buffer handles to be safe to
// retain behind an Arc; mutable access remains mediated by `StorageMut`.
unsafe impl<P, T> Sync for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod + Sync,
{
}

impl<P, T> Storage<T> for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar + bytemuck::Pod,
{
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<P, T> StorageMut<T> for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar + bytemuck::Pod,
{
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) <= 1 {
            return;
        }
        let mut host = vec![T::zero(); self.buffer.len()];
        P::device()
            .download(self.buffer.as_ref(), &mut host)
            .expect("Hephaestus storage uniqueness download failed");
        let replacement = P::device()
            .upload(&host)
            .expect("Hephaestus storage uniqueness upload failed");
        self.buffer = Arc::new(replacement);
    }
}
