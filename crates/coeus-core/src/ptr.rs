// ── Parallel-safe raw pointer wrappers ──
//
// Raw pointers (`*const T`, `*mut T`) are not `Send + Sync` by default,
// so they can't be captured by parallel loop closures. These newtypes
// assert the safety invariants needed for parallel dispatch.

/// Const pointer wrapper with conditional `Send`/`Sync` capability.
#[repr(transparent)]
pub struct SendPtr<T>(pub *const T);
// SAFETY: moving the wrapper transfers only the address; values read through
// it must be safe to move to the receiving thread.
unsafe impl<T: Send> Send for SendPtr<T> {}
// SAFETY: shared reads require the pointee to support concurrent access.
unsafe impl<T: Sync> Sync for SendPtr<T> {}

impl<T> Clone for SendPtr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

impl<T: Copy> SendPtr<T> {
    /// Read element at offset `i` (element count, not bytes).
    ///
    /// # Safety
    /// Caller must ensure that index `i` is within the allocated bounds of the pointer.
    #[inline]
    pub unsafe fn read(&self, i: usize) -> T {
        // SAFETY: The caller guarantees that index `i` is within the allocated bounds of the pointer.
        *self.0.add(i)
    }

    /// Borrow a contiguous range at element offset `start`.
    ///
    /// # Safety
    /// Caller must ensure `start..start + len` is within the allocated bounds
    /// of the pointer and that the returned range is not mutably aliased for
    /// the returned lifetime.
    #[inline]
    pub unsafe fn slice<'a>(&self, start: usize, len: usize) -> &'a [T] {
        // SAFETY: The caller guarantees that the requested range is in bounds
        // and not mutably aliased for the returned lifetime.
        std::slice::from_raw_parts(self.0.add(start), len)
    }
}

/// Mutable pointer wrapper with conditional `Send`/`Sync` capability.
#[repr(transparent)]
pub struct SendPtrMut<T>(pub *mut T);
// SAFETY: values written through the wrapper cross the thread boundary and
// the caller must still prove that mutable accesses are disjoint.
unsafe impl<T: Send> Send for SendPtrMut<T> {}
// SAFETY: sharing the wrapper is sound only when callers uphold the disjoint
// mutable-access contract; written values must be thread-sendable.
unsafe impl<T: Send> Sync for SendPtrMut<T> {}

impl<T> Clone for SendPtrMut<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtrMut<T> {}

impl<T: Copy> SendPtrMut<T> {
    /// Read element at offset `i`.
    ///
    /// # Safety
    /// Caller must ensure that index `i` is within bounds.
    #[inline]
    pub unsafe fn read(&self, i: usize) -> T {
        // SAFETY: The caller guarantees that index `i` is within bounds and the pointer is valid.
        *self.0.add(i)
    }

    /// Write element at offset `i`.
    ///
    /// # Safety
    /// Caller must ensure that index `i` is within bounds.
    #[inline]
    pub unsafe fn write(&self, i: usize, val: T) {
        // SAFETY: The caller guarantees that index `i` is within bounds and the pointer is valid.
        *self.0.add(i) = val;
    }

    /// Borrow a contiguous mutable range at element offset `start`.
    ///
    /// # Safety
    /// Caller must ensure `start..start + len` is within the allocated bounds.
    #[inline]
    pub unsafe fn slice_mut<'a>(&self, start: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}
