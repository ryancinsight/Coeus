// ── Copy-on-Write storage wrapper ──
// Wraps any Storage+StorageMut to provide transparent COW semantics.

use crate::storage::cpu::CpuStorage;
use crate::storage::traits::{
    CpuAddressableStorage, CpuAddressableStorageMut, Storage, StorageMut,
};

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

impl<T> CowStorage<CpuStorage<T>>
where
    T: Copy + Send + Sync + 'static,
{
    /// Returns true when the wrapped CPU storage has exclusive allocation ownership.
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.inner.is_unique()
    }
}

impl<S: Storage<T>, T> Storage<T> for CowStorage<S> {
    type Error = S::Error;

    #[inline]
    fn try_allocate(len: usize) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: S::try_allocate(len)?,
        })
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
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        self.inner.try_as_mut_slice()
    }

    #[inline]
    fn make_unique(&mut self) -> Result<(), Self::Error> {
        self.inner.make_unique()
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
    fn as_mut_slice(&mut self) -> Result<&mut [T], Self::Error> {
        self.inner.as_mut_slice()
    }
}
