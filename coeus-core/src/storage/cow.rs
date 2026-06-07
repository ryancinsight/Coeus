// ── Copy-on-Write storage wrapper ──
// Wraps any Storage+StorageMut to provide transparent COW semantics.

use crate::storage::traits::{Storage, StorageMut, CpuAddressableStorage, CpuAddressableStorageMut};

/// COW wrapper over an inner storage.
///
/// Delays copies until a mutable borrow is taken.
/// The inner storage is responsible for the actual COW logic.
#[repr(transparent)]
#[derive(Clone)]
pub struct CowStorage<S> {
    inner: S,
}

impl<S> crate::storage::traits::private::Sealed for CowStorage<S> {}

impl<S> CowStorage<S> {
    /// Wrap a storage in COW semantics.
    #[inline]
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Unwrap the inner storage.
    #[inline]
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Reference to inner.
    #[inline]
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Mutable reference to inner (no COW — caller manages).
    #[inline]
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: Storage<T>, T> Storage<T> for CowStorage<S> {
    #[inline]
    fn allocate(len: usize) -> Self {
        Self { inner: S::allocate(len) }
    }

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        self.inner.try_as_slice()
    }
}

impl<S: StorageMut<T>, T> StorageMut<T> for CowStorage<S> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        self.inner.try_as_mut_slice()
    }

    #[inline]
    fn make_unique(&mut self) {
        self.inner.make_unique();
    }
}

impl<S: CpuAddressableStorage<T>, T> CpuAddressableStorage<T> for CowStorage<S> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<S: CpuAddressableStorageMut<T>, T> CpuAddressableStorageMut<T> for CowStorage<S> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}
