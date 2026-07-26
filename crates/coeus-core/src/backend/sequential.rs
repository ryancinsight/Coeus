// ── Sequential backend ──
// Zero-sized type for single-threaded execution.

use crate::backend::{Backend, ComputeBackend};
use crate::dtype::Scalar;
use crate::storage::{CpuStorage, Storage};

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

impl crate::backend::traits::private::Sealed for SequentialBackend {}

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

impl ComputeBackend for SequentialBackend {
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
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CpuStorage::allocate(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        use crate::storage::CpuAddressableStorageMut;
        dst.as_mut_slice().fill(val);
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        use crate::storage::CpuAddressableStorageMut;
        dst.as_mut_slice().copy_from_slice(src);
    }

    #[inline]
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        use crate::storage::CpuAddressableStorage;
        dst.copy_from_slice(src.as_slice());
    }
}

impl Backend for SequentialBackend {
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
