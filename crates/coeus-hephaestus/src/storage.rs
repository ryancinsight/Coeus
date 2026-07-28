use crate::{error::HephaestusBackendError, reduction::HephaestusProvider};
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
    /// Allocate zeroed storage and preserve provider failures at the Coeus
    /// backend boundary.
    pub fn try_new(len: usize) -> Result<Self, HephaestusBackendError> {
        let buffer = P::try_device()?
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .map_err(|source| HephaestusBackendError::device("allocate", source))?;
        Ok(Self {
            buffer: Arc::new(buffer),
            marker: PhantomData,
        })
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
    type Error = HephaestusBackendError;

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn try_allocate(len: usize) -> Result<Self, Self::Error> {
        Self::try_new(len)
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
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        Ok(None)
    }

    fn make_unique(&mut self) -> Result<(), Self::Error> {
        if Arc::strong_count(&self.buffer) <= 1 {
            return Ok(());
        }
        let device = P::try_device()?;
        let replacement = device
            .alloc_zeroed_with_hint(
                self.buffer.len(),
                PlacementHint::Tier(self.buffer.tier()),
            )
            .map_err(|source| HephaestusBackendError::device("cow allocate", source))?;
        device
            .copy_buffer(self.buffer.as_ref(), &replacement)
            .map_err(|source| HephaestusBackendError::device("cow copy", source))?;
        self.buffer = Arc::new(replacement);
        Ok(())
    }
}
