// ── ComputeBackend trait ──
// Abstract execution and storage backend interface for heterogenous device computation.

use crate::storage::StorageMut;
// ── Backend trait ──
// Abstract execution backend for parallel and sequential dispatch.

use crate::backend::BackendError;
use crate::dtype::Scalar;

/// Private module for sealing compute backend.
#[doc(hidden)]
pub mod private {
    /// Sealed trait to prevent downstream user implementation.
    pub trait Sealed {}
}

/// General interface for hardware execution backends (CPU, GPU, etc.)
///
/// # Examples
///
/// ```
/// use coeus_core::{ComputeBackend, SequentialBackend};
///
/// let backend = SequentialBackend::new();
/// let mut buf = backend.allocate::<f32>(3).expect("allocation succeeds");
/// backend.fill(&mut buf, 42.0).expect("fill succeeds");
/// let mut host = [0.0_f32; 3];
/// backend.copy_to_host(&buf, &mut host).expect("copy succeeds");
/// assert_eq!(host, [42.0; 3]);
/// ```
pub trait ComputeBackend: private::Sealed + Send + Sync + Clone + 'static {
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

    /// Allocate storage on the device (uninitialized).
    fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error>;

    /// Fill device buffer with a value.
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error>;

    /// Copy data from host (CPU) memory to this device buffer.
    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error>;

    /// Copy data from this device buffer to host (CPU) memory.
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
pub trait Backend: ComputeBackend + Default {
    /// Execute `f(i)` for `i` in `[start, end)` — possibly in parallel.
    ///
    /// The backend decides whether to parallelize (Moirai) or
    /// run sequentially (SequentialBackend).
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static;
}
