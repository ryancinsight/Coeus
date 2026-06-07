// ── ComputeBackend trait ──
// Abstract execution and storage backend interface for heterogenous device computation.

use crate::storage::StorageMut;
// ── Backend trait ──
// Abstract execution backend for parallel and sequential dispatch.

use crate::dtype::Scalar;

/// Private module for sealing compute backend.
#[doc(hidden)]
pub mod private {
    /// Sealed trait to prevent downstream user implementation.
    pub trait Sealed {}
}

/// General interface for hardware execution backends (CPU, GPU, etc.)
pub trait ComputeBackend: private::Sealed + Send + Sync + Clone + 'static {
    /// Memory handle type representing device-allocated storage.
    type DeviceBuffer<T: Scalar>: StorageMut<T>;

    /// Descriptor / configuration params needed for launching/compiling pipelines on this backend.
    type KernelDescriptor;

    /// Async execution handle for non-blocking queue operations.
    type DispatchFuture<T: Scalar>: std::future::Future<Output = T> + Send;

    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Number of workers/threads.
    fn num_threads(&self) -> usize;

    /// Allocate storage on the device (uninitialized).
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T>;

    /// Fill device buffer with a value.
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T);

    /// Copy data from host (CPU) memory to this device buffer.
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>);

    /// Copy data from this device buffer to host (CPU) memory.
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]);
}

/// Trait for backend execution engines.
///
/// # Design
/// - ZST implementations (MoiraiBackend, SequentialBackend)
/// - Monomorphized: `parallel_for` takes a generic closure, not a trait object
/// - The closure `F` is `Fn(usize) + Send + Sync + 'static` for thread safety
pub trait Backend: ComputeBackend + Default {
    /// Execute `f(i)` for `i` in `[start, end)` — possibly in parallel.
    ///
    /// The backend decides whether to parallelize (Moirai) or
    /// run sequentially (SequentialBackend).
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static;
}
