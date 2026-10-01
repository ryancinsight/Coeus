// ── Sequential backend ──
// Zero-sized type for single-threaded execution.

use crate::backend::{Backend, ComputeBackend};
use crate::dtype::Scalar;
use crate::storage::CpuStorage;

/// Sequential (single-threaded) backend.
///
/// # ZST
/// Zero-sized type — used as a compile-time default or fallback.
///
/// # Examples
///
/// ```
/// use coeus_core::{Backend, ComputeBackend, SequentialBackend};
///
/// let backend = SequentialBackend::new();
/// assert_eq!(backend.num_threads(), 1);
/// assert_eq!(backend.name(), "sequential");
///
/// backend.parallel_for(0, 4, |i| {
///     // executes sequentially: 0, 1, 2, 3
/// });
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialBackend;

impl SequentialBackend {
    /// Create a new handle (ZST).
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::{ComputeBackend, SequentialBackend};
    ///
    /// let backend = SequentialBackend::new();
    /// assert_eq!(backend.name(), "sequential");
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: CPU storage tracks initialization, refuses reads while uninitialized,
// and the fill/copy methods fully write before reporting success.
unsafe impl ComputeBackend for SequentialBackend {
    type Error = crate::backend::BackendError;
    type DeviceBuffer<T: Scalar> = CpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "sequential"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        CpuStorage::allocate_uninitialized(len)
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        CpuStorage::filled(len, T::zero())
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error> {
        dst.fill_cow(val)
    }

    #[inline]
    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        dst.copy_from_slice_cow(src)
    }

    #[inline]
    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        use crate::storage::CpuAddressableStorage;
        let source_len = src.as_slice().len();
        if source_len != dst.len() {
            return Err(crate::backend::BackendError::BufferLengthMismatch {
                operation: "copy_to_host",
                source_len,
                destination_len: dst.len(),
            });
        }
        dst.copy_from_slice(src.as_slice());
        Ok(())
    }
}

// SAFETY: the loop invokes every closure call inline before returning.
unsafe impl Backend for SequentialBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        for i in start..end {
            f(i);
        }
    }
}
