/// Private module for sealing storage traits to prevent external implementations,
/// guaranteeing memory safety and monomorphization invariants.
#[doc(hidden)]
pub mod private {
    /// Sealed trait to prevent downstream user implementation.
    pub trait Sealed {}
}

/// Immutable storage access.
///
/// # Examples
///
/// ```
/// use coeus_core::{CpuStorage, Storage};
///
/// let s = CpuStorage::<f32>::try_from_slice(&[1.0, 2.0, 3.0]).expect("allocation succeeds");
/// assert_eq!(s.len(), 3);
/// assert!(!s.is_empty());
/// ```
pub trait Storage<T>: private::Sealed + Clone + Send + Sync + 'static {
    /// Typed failure returned by allocation or storage operations.
    type Error: std::error::Error + From<crate::BackendError> + Send + Sync + 'static;

    /// Number of elements stored.
    fn len(&self) -> usize;

    /// True if storage is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Allocate new storage for `len` elements statically (contents unspecified).
    fn try_allocate(len: usize) -> Result<Self, Self::Error>;

    /// Borrow data as a host CPU slice if addressable.
    fn try_as_slice(&self) -> Option<&[T]>;
}

/// Mutable storage access.
pub trait StorageMut<T>: Storage<T> {
    /// Mutably borrow data as a host CPU slice if addressable.
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error>;

    /// Make the storage allocation unique, triggering Copy-On-Write if shared.
    fn make_unique(&mut self) -> Result<(), Self::Error>;
}

/// Sub-trait for storages that are readable in CPU host memory.
pub trait CpuAddressableStorage<T>: Storage<T> {
    /// Borrow data as a contiguous slice.
    fn as_slice(&self) -> &[T];
}

/// Sub-trait for storages that are mutable in CPU host memory.
///
/// # COW semantics
/// Implementors MUST perform copy-on-write if the underlying
/// allocation is shared.
pub trait CpuAddressableStorageMut<T>: StorageMut<T> + CpuAddressableStorage<T> {
    /// Mutably borrow data as a contiguous slice.
    ///
    /// Triggers COW if buffer is shared.
    fn as_mut_slice(&mut self) -> Result<&mut [T], Self::Error>;
}
