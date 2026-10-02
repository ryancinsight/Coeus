use std::alloc::{GlobalAlloc, Layout as AllocLayout};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::storage::{CpuAddressableStorage, CpuAddressableStorageMut, Storage, StorageMut};

/// A single aligned memory block from Mnemosyne.
struct RawBlock {
    ptr: *mut u8,
    layout: AllocLayout,
}

impl RawBlock {
    #[inline]
    fn new(size: usize, align: usize) -> Result<Option<Self>, std::alloc::LayoutError> {
        let layout = AllocLayout::from_size_align(size, align)?;
        if size == 0 {
            return Ok(Some(Self {
                ptr: std::ptr::without_provenance_mut(align),
                layout,
            }));
        }
        // SAFETY: `layout` is constructed with valid non-zero size and alignment verified by AllocLayout::from_size_align.
        let ptr = unsafe { mnemosyne::Mnemosyne.alloc(layout) };
        if ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(Self { ptr, layout }))
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    #[inline]
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for RawBlock {
    #[inline]
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            // SAFETY: `self.ptr` is a non-null, valid pointer previously allocated by Mnemosyne with the exact same `self.layout`.
            unsafe {
                mnemosyne::Mnemosyne.dealloc(self.ptr, self.layout);
            }
        }
    }
}

// SAFETY: moving ownership does not change the allocation address; Mnemosyne
// permits deallocation on another thread. Typed access is confined to CpuStorage.
unsafe impl Send for RawBlock {}
// SAFETY: shared block references expose no safe memory access. CpuStorage
// allows shared typed reads and detaches the allocation before mutation.
unsafe impl Sync for RawBlock {}

/// CPU-side aligned buffer with COW semantics via `Arc`.
///
/// Built on Mnemosyne for allocation. Cloning is `Arc::clone` (cheap).
/// Mutation on a shared buffer triggers a deep copy (COW).
///
/// # Examples
///
/// Allocate, fill, and read back data:
///
/// ```
/// use coeus_core::CpuStorage;
/// use coeus_core::storage::CpuAddressableStorage;
///
/// let mut s = CpuStorage::<f32>::new(4).expect("CPU storage allocation succeeds");
/// let slice = s.as_slice();
/// assert_eq!(slice.len(), 4);
/// ```
///
/// Clone is cheap (Arc refcount); mutation triggers COW:
///
/// ```
/// use coeus_core::CpuStorage;
/// use coeus_core::storage::{CpuAddressableStorage, CpuAddressableStorageMut};
///
/// let a = CpuStorage::<f32>::from_slice(&[1.0, 2.0, 3.0])?;
/// let b = a.clone();       // Arc clone — no data copy
/// assert!(!a.is_unique());  // shared
///
/// let mut c = a.clone();
/// c.as_mut_slice()?[0] = 99.0; // COW: deep copy happens here
/// assert!(c.is_unique());     // now unique after mutation
/// assert_eq!(b.as_slice()[0], 1.0); // original unchanged
/// # Ok::<(), coeus_core::BackendError>(())
/// ```
///
/// Allocation ownership stays private: safe code cannot replace the pointer
/// used by destruction with another live allocation's pointer.
///
/// ```compile_fail
/// use coeus_core::CpuStorage;
///
/// let mut victim = CpuStorage::filled(1, 7_u8).expect("CPU storage allocation succeeds");
/// let mut owner = CpuStorage::filled(1, 0_u8).into_raw().unwrap().expect("CPU storage allocation succeeds");
/// owner.ptr = victim.raw_slice_mut_cow().expect("invariant: test allocation succeeds").as_mut_ptr();
/// drop(owner);
/// victim.raw_slice_mut_cow().expect("invariant: test allocation succeeds")[0] = 9;
/// ```
#[derive(Clone)]
pub struct CpuStorage<T> {
    block: Arc<RawBlock>,
    len: usize,
    initialized: bool,
    _marker: PhantomData<T>,
}

impl<T> crate::storage::traits::private::Sealed for CpuStorage<T> {}

// SAFETY: initialized elements may cross threads when T is Send; shared
// allocation ownership is synchronized by Arc and mutation detaches it.
unsafe impl<T: Send> Send for CpuStorage<T> {}
// SAFETY: shared access reads initialized T values only; mutable access requires
// an exclusive storage borrow and detaches any shared allocation first.
unsafe impl<T: Sync> Sync for CpuStorage<T> {}

impl<T: Copy + Send + Sync + 'static> CpuStorage<T> {
    /// Allocate an uninitialized buffer while preserving allocation failures.
    pub(crate) fn allocate_uninitialized(len: usize) -> Result<Self, crate::BackendError> {
        let byte_size =
            len.checked_mul(std::mem::size_of::<T>())
                .ok_or(crate::BackendError::Overflow {
                    operation: "cpu allocation",
                    reason: "element-count byte-size overflow",
                })?;
        let align = std::mem::align_of::<T>();
        let block = RawBlock::new(byte_size, align)
            .map_err(|_| crate::BackendError::Overflow {
                operation: "cpu allocation",
                reason: "allocation layout exceeds addressable size",
            })?
            .ok_or(crate::BackendError::AllocatorExhausted {
                operation: "cpu allocation",
            })?;
        Ok(Self {
            block: Arc::new(block),
            len,
            initialized: len == 0,
            _marker: PhantomData,
        })
    }

    /// Allocate a new zero-initialized buffer for `len` elements of type `T`.
    ///
    /// # Errors
    /// Returns the allocator error when the requested byte size overflows or
    /// the provider cannot allocate the storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::CpuStorage;
    /// use coeus_core::storage::{CpuAddressableStorage, Storage};
    ///
    /// let s = CpuStorage::<f64>::new(8)?;
    /// assert_eq!(s.len(), 8);
    /// # Ok::<(), coeus_core::BackendError>(())
    /// ```
    #[inline]
    pub fn new(len: usize) -> Result<Self, crate::BackendError>
    where
        T: crate::Scalar,
    {
        Self::filled(len, T::zero())
    }

    /// Allocate and initialize every element with `value` without first
    /// constructing a typed slice over uninitialized memory.
    #[inline]
    pub fn filled(len: usize, value: T) -> Result<Self, crate::BackendError> {
        let mut storage = Self::allocate_uninitialized(len)?;
        let destination = storage.block.as_mut_ptr().cast::<T>();
        for index in 0..len {
            // SAFETY: `destination` is aligned and allocated for `len`
            // elements. Each index is written exactly once before `storage`
            // is returned through its safe readable API.
            unsafe { destination.add(index).write(value) };
        }
        storage.initialized = true;
        Ok(storage)
    }

    /// Create from existing slice (copies data).
    ///
    /// # Errors
    /// Returns the allocator error when the provider cannot allocate storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::CpuStorage;
    /// use coeus_core::storage::CpuAddressableStorage;
    ///
    /// let s = CpuStorage::from_slice(&[1.0_f32, 2.0, 3.0])?;
    /// assert_eq!(s.as_slice(), &[1.0, 2.0, 3.0]);
    /// # Ok::<(), coeus_core::BackendError>(())
    /// ```
    #[inline]
    pub fn from_slice(data: &[T]) -> Result<Self, crate::BackendError> {
        let mut storage = Self::allocate_uninitialized(data.len())?;
        // SAFETY: the destination is aligned and allocated for `data.len()`
        // elements, the source is a valid non-overlapping slice, and `T:
        // Copy` needs no per-element drop handling. The whole destination is
        // initialized before the storage escapes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                storage.block.as_mut_ptr().cast::<T>(),
                data.len(),
            );
        }
        storage.initialized = true;
        Ok(storage)
    }

    /// Returns true when this storage has exclusive ownership of its allocation.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.block) == 1
    }

    fn detach_shared_allocation(&mut self) -> Result<(), crate::BackendError> {
        if self.is_unique() {
            return Ok(());
        }

        let mut replacement = Self::allocate_uninitialized(self.len)?;
        // SAFETY: `MaybeUninit<T>` accepts every bit pattern. Both allocations
        // have the same alignment and `len`-element extent, and the source and
        // destination blocks are distinct. This copies initialized and
        // uninitialized elements without forming a reference to either.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.block.as_ptr().cast::<std::mem::MaybeUninit<T>>(),
                replacement
                    .block
                    .as_mut_ptr()
                    .cast::<std::mem::MaybeUninit<T>>(),
                self.len,
            );
        }
        replacement.initialized = self.initialized;
        *self = replacement;
        Ok(())
    }

    /// Fill storage directly, including an uninitialized allocation.
    pub(crate) fn fill_cow(&mut self, value: T) -> Result<(), crate::BackendError> {
        self.detach_shared_allocation()?;
        let destination = self.block.as_mut_ptr().cast::<T>();
        for index in 0..self.len {
            // SAFETY: the block is aligned and allocated for `self.len`
            // elements, and each element is overwritten before storage is
            // marked initialized.
            unsafe { destination.add(index).write(value) };
        }
        self.initialized = true;
        Ok(())
    }

    /// Copy a complete host slice into storage without reading its old bytes.
    pub(crate) fn copy_from_slice_cow(&mut self, source: &[T]) -> Result<(), crate::BackendError> {
        if source.len() != self.len {
            return Err(crate::BackendError::BufferLengthMismatch {
                operation: "copy_to_device",
                source_len: source.len(),
                destination_len: self.len,
            });
        }
        self.detach_shared_allocation()?;
        // SAFETY: the destination is aligned and allocated for `self.len`
        // elements; the source is a valid slice of the same length; COW made
        // the destination allocation unique; and `T: Copy` permits bitwise
        // copying. The full destination is initialized before the flag changes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                self.block.as_mut_ptr().cast::<T>(),
                self.len,
            );
        }
        self.initialized = true;
        Ok(())
    }

    fn initialize_zeroed(&mut self)
    where
        T: crate::Scalar,
    {
        let destination = self.block.as_mut_ptr().cast::<T>();
        for index in 0..self.len {
            // SAFETY: the block is aligned and allocated for `self.len`
            // elements, and each element receives a valid scalar value.
            unsafe { destination.add(index).write(T::zero()) };
        }
        self.initialized = true;
    }

    #[inline]
    fn raw_slice(&self) -> &[T] {
        assert!(
            self.initialized,
            "invariant: CPU storage is initialized before reading"
        );
        // SAFETY: The underlying block pointer is aligned, valid, non-null,
        // initialized, and allocated for `self.len` elements of type `T`.
        unsafe { std::slice::from_raw_parts(self.block.as_ptr() as *const T, self.len) }
    }

    /// Mutable raw slice — bypasses COW. Unsafe.
    ///
    /// # Safety
    /// Caller must ensure that self has exclusive, unique access and is the sole owner.
    #[inline]
    unsafe fn raw_slice_mut(&mut self) -> &mut [T] {
        assert!(
            self.initialized,
            "invariant: CPU storage is initialized before mutable borrowing"
        );
        // SAFETY: The block pointer is aligned, valid, non-null, initialized,
        // and allocated for `self.len` elements. The mutable borrow and COW
        // guarantee exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.block.as_mut_ptr() as *mut T, self.len) }
    }

    /// Mutable raw slice with COW handling.
    ///
    /// # Errors
    ///
    /// Returns an allocation error if detaching the shared block fails.
    #[inline]
    pub fn raw_slice_mut_cow(&mut self) -> Result<&mut [T], crate::BackendError>
    where
        T: crate::Scalar,
    {
        self.detach_shared_allocation()?;
        if !self.initialized {
            self.initialize_zeroed();
        }
        // SAFETY: COW guarantees unique ownership, and uninitialized elements
        // are zeroed before a typed mutable slice is formed.
        Ok(unsafe { self.raw_slice_mut() })
    }
}

impl<T: crate::Scalar> Storage<T> for CpuStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        self.initialized.then(|| self.raw_slice())
    }
}

impl<T: crate::Scalar> StorageMut<T> for CpuStorage<T> {
    type Error = crate::BackendError;

    #[inline]
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        self.raw_slice_mut_cow().map(Some)
    }

    #[inline]
    fn make_unique(&mut self) -> Result<(), Self::Error> {
        self.detach_shared_allocation()
    }
}

impl<T: crate::Scalar> CpuAddressableStorage<T> for CpuStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.raw_slice()
    }
}

impl<T: crate::Scalar> CpuAddressableStorageMut<T> for CpuStorage<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> Result<&mut [T], Self::Error> {
        self.raw_slice_mut_cow()
    }
}

#[cfg(test)]
mod tests {
    use super::CpuStorage;
    use crate::BackendError;

    #[test]
    fn allocation_size_overflow_is_reported() {
        let Err(error) = CpuStorage::<u16>::allocate_uninitialized(usize::MAX) else {
            panic!("an overflowing allocation size must fail");
        };

        assert!(matches!(
            error,
            BackendError::Overflow {
                operation: "cpu allocation",
                reason: "element-count byte-size overflow",
            }
        ));
    }

    #[test]
    fn uninitialized_storage_cannot_be_read_and_mutable_access_initializes_it() {
        use crate::storage::{CpuAddressableStorage, CpuAddressableStorageMut, Storage};

        let storage = CpuStorage::<u32>::allocate_uninitialized(3)
            .expect("invariant: small CPU storage allocation succeeds");
        assert!(Storage::try_as_slice(&storage).is_none());

        let mut shared = storage.clone();
        let values = CpuAddressableStorageMut::as_mut_slice(&mut shared)
            .expect("invariant: COW allocation succeeds");
        assert_eq!(values, &[0, 0, 0]);
        values.copy_from_slice(&[5, 8, 13]);

        assert_eq!(CpuAddressableStorage::as_slice(&shared), &[5, 8, 13]);
        assert!(Storage::try_as_slice(&storage).is_none());
    }

    #[test]
    fn empty_uninitialized_storage_is_readable() {
        use crate::storage::{CpuAddressableStorage, Storage};

        let storage = CpuStorage::<u32>::allocate_uninitialized(0)
            .expect("invariant: empty CPU storage allocation succeeds");
        assert_eq!(Storage::try_as_slice(&storage), Some([].as_slice()));
        assert_eq!(CpuAddressableStorage::as_slice(&storage), &[] as &[u32]);
    }

    #[test]
    fn complete_fill_and_copy_initialize_uninitialized_storage() {
        use crate::storage::CpuAddressableStorage;

        let mut filled = CpuStorage::<u16>::allocate_uninitialized(2)
            .expect("invariant: small CPU storage allocation succeeds");
        filled
            .fill_cow(7)
            .expect("invariant: unique fill does not allocate");
        assert_eq!(filled.as_slice(), &[7, 7]);

        let mut copied = CpuStorage::<u16>::allocate_uninitialized(3)
            .expect("invariant: small CPU storage allocation succeeds");
        copied
            .copy_from_slice_cow(&[2, 3, 5])
            .expect("invariant: unique copy does not allocate");
        assert_eq!(copied.as_slice(), &[2, 3, 5]);
    }

    #[test]
    fn public_filled_reports_allocation_size_overflow() {
        let Err(error) = CpuStorage::<u16>::filled(usize::MAX, 0) else {
            panic!("an overflowing public allocation must fail");
        };

        assert!(matches!(
            error,
            BackendError::Overflow {
                operation: "cpu allocation",
                reason: "element-count byte-size overflow",
            }
        ));
    }
}
