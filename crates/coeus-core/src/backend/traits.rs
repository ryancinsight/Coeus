// ── ComputeBackend trait ──
// Abstract execution and storage backend interface for heterogenous device computation.

use crate::storage::StorageMut;
// ── Backend trait ──
// Abstract execution backend for parallel and sequential dispatch.

use crate::backend::BackendError;
use crate::dtype::Scalar;

/// General interface for hardware execution backends (CPU, GPU, etc.)
///
/// The trait is deliberately open. Its implementor set spans sibling crates —
/// one per accelerator vendor — so a seal would make the prescribed
/// cross-crate implementations uncompilable.
///
/// # Examples
///
/// ```
/// use coeus_core::{ComputeBackend, SequentialBackend};
///
/// let backend = SequentialBackend::new();
/// let mut buf = backend.allocate_zeroed::<f32>(3)?;
/// backend.fill(&mut buf, 42.0)?;
/// let mut host = [0.0_f32; 3];
/// backend.copy_to_host(&buf, &mut host)?;
/// assert_eq!(host, [42.0; 3]);
/// # Ok::<(), coeus_core::BackendError>(())
/// ```
///
/// # Safety
/// Implementations must keep `DeviceBuffer`'s safe accessors sound. In
/// particular, `allocate_zeroed`, `fill`, `fill_zero`, and
/// `copy_to_device` initialize every element on success; `try_as_slice` must
/// not expose uninitialized elements; and failed operations must not make a
/// previously initialized buffer invalid. The caller of `allocate` must
/// initialize every element before invoking any operation that reads it.
pub unsafe trait ComputeBackend: Send + Sync + Clone + 'static {
    /// Typed failure returned by fallible backend operation traits.
    type Error: std::error::Error + From<BackendError> + Send + Sync + 'static;

    /// Memory handle type representing device-allocated storage.
    type DeviceBuffer<T: Scalar>: StorageMut<T, Error = Self::Error>;

    /// Descriptor / configuration params needed for launching/compiling pipelines on this backend.
    type KernelDescriptor;

    /// Async execution handle for non-blocking queue operations.
    type DispatchFuture<T: Scalar>: std::future::Future<Output = T> + Send;

    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Number of workers/threads.
    fn num_threads(&self) -> usize;

    /// Allocate storage on the device without initializing its elements.
    ///
    /// # Safety
    /// The caller must initialize every element before any operation reads the
    /// returned buffer. An implementation must provide storage that is valid
    /// for `len` elements and must not expose uninitialized elements through a
    /// safe read operation.
    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error>;

    /// Allocate zero-initialized storage on the device.
    ///
    /// Backends with native zeroed allocation should override this method so
    /// construction does not require a separate fill pass.
    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // SAFETY: `fill_zero` initializes every element before `dst` is
        // returned to the caller.
        let mut dst = unsafe { self.allocate(len)? };
        self.fill_zero(&mut dst)?;
        Ok(dst)
    }

    /// Fill device buffer with a value.
    ///
    /// Other storage clones retain their values when this buffer is shared.
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error>;

    /// Fill a device buffer with the additive identity.
    ///
    /// Accelerator backends override this method with their native clear or
    /// memset operation, avoiding destination-sized host staging.
    /// Other storage clones retain their values when this buffer is shared.
    #[inline]
    fn fill_zero<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>) -> Result<(), Self::Error> {
        self.fill(dst, T::zero())
    }

    /// Copy data from host (CPU) memory to this device buffer.
    ///
    /// Other storage clones retain their values when this buffer is shared.
    /// The source length must equal the destination length; an invalid length
    /// is rejected before the destination is detached or written.
    ///
    /// # Errors
    /// Returns a typed error when lengths differ, storage detachment fails, or
    /// the backend rejects the transfer.
    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error>;

    /// Copy data from this device buffer to host (CPU) memory.
    /// The destination length must equal the source length; an invalid length
    /// is rejected before the destination is written.
    ///
    /// # Errors
    /// Returns a typed error when lengths differ or the backend rejects the
    /// transfer.
    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error>;
}

/// Trait for backend execution engines.
///
/// # Design
/// - ZST implementations (MoiraiBackend, SequentialBackend)
/// - Monomorphized: `parallel_for` takes a generic closure, not a trait object
/// - The closure `F` is `Fn(usize) + Send + Sync + 'static` for thread safety
///
/// # Examples
///
/// ```
/// use coeus_core::{Backend, SequentialBackend};
///
/// let backend = SequentialBackend::new();
/// let mut sum = 0usize;
/// backend.parallel_for(0, 5, |i| {
///     // In a real kernel this would write to a pre-allocated output slice.
///     // SequentialBackend executes in order: 0, 1, 2, 3, 4.
/// });
/// ```
/// # Safety
///
/// If [`Backend::parallel_for`] returns normally, implementations must invoke
/// the supplied closure exactly once for each index in `[start, end)` and for
/// no other index. They must not return until every invocation has completed.
/// If the closure panics, implementations must join any in-flight invocations
/// before unwinding. CPU kernels rely on these guarantees for disjoint writes
/// into uninitialized output storage.
pub unsafe trait Backend: ComputeBackend + Default {
    /// Execute `f(i)` for `i` in `[start, end)` — possibly in parallel.
    ///
    /// The backend decides whether to parallelize (Moirai) or
    /// run sequentially (SequentialBackend).
    /// This method returns only after every invocation of `f` has completed.
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static;
}

#[cfg(test)]
mod tests {
    use super::{Backend, ComputeBackend};
    use crate::{
        BackendError, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend,
        SequentialBackend,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn assert_backend_visits_each_index_once<B: Backend>(backend: B, len: usize) {
        let start = 3;
        let end = start + len;
        let visits = Arc::new((0..end).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let worker_visits = Arc::clone(&visits);
        backend.parallel_for(start, end, move |index| {
            worker_visits[index].fetch_add(1, Ordering::Relaxed);
        });

        assert!(visits[..start]
            .iter()
            .all(|count| count.load(Ordering::Relaxed) == 0));
        for index in start..end {
            assert_eq!(
                visits[index].load(Ordering::Relaxed),
                1,
                "parallel_for must visit index {index} exactly once"
            );
        }
    }

    fn assert_copy_length_errors<B>()
    where
        B: ComputeBackend<Error = BackendError> + Default,
        B::DeviceBuffer<u32>: CpuAddressableStorage<u32> + CpuAddressableStorageMut<u32>,
    {
        let backend = B::default();
        let mut storage = backend
            .allocate_zeroed::<u32>(2)
            .expect("invariant: two-element CPU buffer allocation succeeds");
        backend
            .copy_to_device(&[7, 11], &mut storage)
            .expect("invariant: matching host-to-device copy succeeds");

        let error = match backend.copy_to_device(&[13], &mut storage) {
            Err(error) => error,
            Ok(()) => panic!("short host input must be rejected"),
        };
        assert!(matches!(
            error,
            BackendError::BufferLengthMismatch {
                operation: "copy_to_device",
                source_len: 1,
                destination_len: 2,
            }
        ));

        let mut retained = [0; 2];
        backend
            .copy_to_host(&storage, &mut retained)
            .expect("invariant: matching device-to-host copy succeeds");
        assert_eq!(retained, [7, 11]);

        let mut short_output = [19];
        let error = match backend.copy_to_host(&storage, &mut short_output) {
            Err(error) => error,
            Ok(()) => panic!("short host output must be rejected"),
        };
        assert!(matches!(
            error,
            BackendError::BufferLengthMismatch {
                operation: "copy_to_host",
                source_len: 2,
                destination_len: 1,
            }
        ));
        assert_eq!(short_output, [19]);
    }

    #[test]
    fn sequential_backend_rejects_mismatched_transfer_lengths_without_writes() {
        assert_copy_length_errors::<SequentialBackend>();
    }

    #[test]
    fn moirai_backend_rejects_mismatched_transfer_lengths_without_writes() {
        assert_copy_length_errors::<MoiraiBackend>();
    }

    #[test]
    fn sequential_backend_visits_each_parallel_index_once() {
        assert_backend_visits_each_index_once(SequentialBackend::new(), 13);
    }

    #[test]
    fn moirai_backend_visits_each_parallel_index_once() {
        assert_backend_visits_each_index_once(MoiraiBackend::new(), 4_097);
    }
}
