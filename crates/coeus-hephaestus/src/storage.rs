use crate::reduction::HephaestusProvider;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use std::{marker::PhantomData, sync::Arc};
use themis::{MemoryTier, PlacementHint};

/// Identity of a live provider allocation for reuse diagnostics.
///
/// Addresses can be reused after deallocation; retain the compared storage
/// owners while comparing their identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationId(*const ());

/// Reference-counted Coeus storage backed by one Hephaestus device buffer.
pub struct HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod,
{
    buffer: Arc<<P::Device as ComputeDevice>::Buffer<T>>,
    marker: PhantomData<P>,
}

impl<P, T> Clone for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod,
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
    T: Scalar,
{
    /// Adopt an initialized provider buffer without copying its contents.
    #[must_use]
    pub fn from_buffer(buffer: <P::Device as ComputeDevice>::Buffer<T>) -> Self {
        Self {
            buffer: Arc::new(buffer),
            marker: PhantomData,
        }
    }

    /// Allocate zeroed storage in the provider's device tier.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let buffer = P::device()
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Hephaestus provider allocation failed");
        Self::from_buffer(buffer)
    }

    pub(crate) fn uninitialized(len: usize) -> Self {
        let buffer = P::device()
            .alloc_uninitialized_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Hephaestus provider allocation failed");
        Self::from_buffer(buffer)
    }

    /// Identify the allocation without exposing its reference-counted owner.
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        AllocationId(Arc::as_ptr(&self.buffer).cast())
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
    T: eunomia::Pod,
{
}

// SAFETY: `HephaestusProvider` requires its device buffers to be safe to move
// between threads while the provider owns the device synchronization contract.
unsafe impl<P, T> Send for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod + Send,
{
}

// SAFETY: `HephaestusProvider` requires shared buffer handles to be safe to
// retain behind an Arc; mutable access remains mediated by `StorageMut`.
unsafe impl<P, T> Sync for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod + Sync,
{
}

impl<P, T> Storage<T> for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar,
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
    T: Scalar,
{
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) <= 1 {
            return;
        }
        // COW detachment is a storage operation, so preserve the provider's
        // allocation tier and keep the full payload on-device. The device
        // copy overwrites every element before the detached buffer is exposed,
        // so the replacement does not require a redundant initialization pass.
        // The `StorageMut` contract is infallible; provider failures therefore
        // panic until that upstream contract propagates typed failures.
        let device = P::device();
        let replacement = device
            .alloc_uninitialized_with_hint(
                self.buffer.len(),
                PlacementHint::Tier(self.buffer.tier()),
            )
            .expect("Hephaestus storage uniqueness allocation failed");
        device
            .copy_buffer(self.buffer.as_ref(), &replacement)
            .expect("Hephaestus storage uniqueness device copy failed");
        self.buffer = Arc::new(replacement);
    }
}

#[cfg(test)]
mod tests;
