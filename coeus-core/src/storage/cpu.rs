// ── Mnemosyne-backed CPU storage ──
// Reference-counted, aligned allocation using the Mnemosyne allocator.

use std::alloc::{GlobalAlloc, Layout as AllocLayout};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::storage::{Storage, StorageMut, CpuAddressableStorage, CpuAddressableStorageMut};

// ── Aligned raw block ──

/// A single aligned memory block from Mnemosyne.
pub struct RawBlock {
    pub ptr: *mut u8,
    pub layout: AllocLayout,
}

impl RawBlock {
    #[inline]
    fn new(size: usize, align: usize) -> Option<Self> {
        let layout = AllocLayout::from_size_align(size, align).ok()?;
        if size == 0 {
            return Some(Self { ptr: std::ptr::null_mut(), layout });
        }
        // SAFETY: `layout` is constructed with valid non-zero size and alignment verified by AllocLayout::from_size_align.
        let ptr = unsafe { mnemosyne::Mnemosyne.alloc(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, layout })
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 { self.ptr }
    #[inline]
    fn as_mut_ptr(&self) -> *mut u8 { self.ptr }
}

impl Drop for RawBlock {
    #[inline]
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            // SAFETY: `self.ptr` is a non-null, valid pointer previously allocated by Mnemosyne with the exact same `self.layout`.
            unsafe { mnemosyne::Mnemosyne.dealloc(self.ptr, self.layout); }
        }
    }
}

unsafe impl Send for RawBlock {}
unsafe impl Sync for RawBlock {}

// ── CpuStorage ──

/// CPU-side aligned buffer with COW semantics via `Arc`.
///
/// Built on Mnemosyne for allocation. Cloning is `Arc::clone` (cheap).
/// Mutation on a shared buffer triggers a deep copy (COW).
#[derive(Clone)]
pub struct CpuStorage<T> {
    block: Arc<RawBlock>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> crate::storage::traits::private::Sealed for CpuStorage<T> {}

unsafe impl<T: Send> Send for CpuStorage<T> {}
unsafe impl<T: Sync> Sync for CpuStorage<T> {}

impl<T: Copy + Send + Sync + 'static> CpuStorage<T> {
    /// Allocate a new buffer for `len` elements of type `T`.
    ///
    /// # Panics
    /// If Mnemosyne allocation fails.
    #[inline]
    pub fn new(len: usize) -> Self {
        let byte_size = len * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let block = RawBlock::new(byte_size, align)
            .expect("Mnemosyne allocation failed in CpuStorage");
        Self {
            block: Arc::new(block),
            len,
            _marker: PhantomData,
        }
    }

    /// Create from existing slice (copies data).
    #[inline]
    pub fn from_slice(data: &[T]) -> Self {
        let mut s = Self::new(data.len());
        s.raw_slice_mut_cow().copy_from_slice(data);
        s
    }

    /// Consume and return the underlying raw block.
    #[inline]
    pub fn into_raw(self) -> Option<RawBlock> {
        Arc::try_unwrap(self.block).ok()
    }

    /// Returns true when this storage has exclusive ownership of its allocation.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.block) == 1
    }

    // ── Internal helpers ──
    #[inline]
    fn raw_slice(&self) -> &[T] {
        // SAFETY: The underlying block pointer is aligned, valid, and non-null (allocated via Mnemosyne) for `self.len` elements of type `T`.
        unsafe {
            std::slice::from_raw_parts(
                self.block.as_ptr() as *const T,
                self.len,
            )
        }
    }

    /// Mutable raw slice — bypasses COW. Unsafe.
    ///
    /// # Safety
    /// Caller must ensure that self has exclusive, unique access and is the sole owner.
    #[inline]
    unsafe fn raw_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: The block pointer is aligned, valid, and non-null for `self.len` elements, and the mutable borrow guarantees exclusive access.
        std::slice::from_raw_parts_mut(
            self.block.as_mut_ptr() as *mut T,
            self.len,
        )
    }

    /// Mutable raw slice with COW handling.
    #[inline]
    pub fn raw_slice_mut_cow(&mut self) -> &mut [T] {
        if Arc::strong_count(&self.block) > 1 {
            let old_slice = self.raw_slice();
            let mut new_storage = Self::new(self.len);
            // SAFETY: `new_storage` is a newly allocated, unique storage block, so writing to it is safe, and `old_slice` points to a valid memory block.
            unsafe {
                new_storage.raw_slice_mut().copy_from_slice(old_slice);
            }
            *self = new_storage;
        }
        // SAFETY: The strong count check and potential copy-on-write reallocation guarantee that we hold the unique reference to the memory block, making mutable slicing safe.
        unsafe { self.raw_slice_mut() }
    }
}


impl<T: Copy + Send + Sync + 'static> Storage<T> for CpuStorage<T> {
    #[inline]
    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.raw_slice())
    }
}

impl<T: Copy + Send + Sync + 'static> StorageMut<T> for CpuStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(self.raw_slice_mut_cow())
    }

    #[inline]
    fn make_unique(&mut self) {
        self.raw_slice_mut_cow();
    }
}

impl<T: Copy + Send + Sync + 'static> CpuAddressableStorage<T> for CpuStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.raw_slice()
    }
}

impl<T: Copy + Send + Sync + 'static> CpuAddressableStorageMut<T> for CpuStorage<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.raw_slice_mut_cow()
    }
}
