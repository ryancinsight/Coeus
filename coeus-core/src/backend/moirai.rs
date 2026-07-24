// ── Moirai backend ──
// Zero-sized type dispatching work to the Moirai work-stealing engine.

use crate::backend::{Backend, ComputeBackend};
use crate::dtype::Scalar;
use crate::storage::{CpuStorage, Storage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Cached `available_parallelism()` snapshot.
///
/// `std::thread::available_parallelism()` executes a syscall per invocation
/// on most platforms (Linux reads cgroup limits; Windows queries the
/// process affinity mask). Hot-path callers — the conv1d/conv2d/conv3d
/// `Backend` kernels — invoke `num_threads()` per kernel call to decide
/// between the parallel row-partition and the sequential short-circuit.
/// Cache the result once with a relaxed-atomic snapshot so subsequent
/// reads are lock-free and syscall-free.
///
/// # Invariant
/// `available_parallelism()` is monotonic within a process — the kernel
/// may advertise fewer cores at boot via affinity adjustments but never
/// advertises more. A single snapshot therefore remains correct for the
/// entire process lifetime; the relaxed load returns the value observed
/// on the first call.
static AVAILABLE_PARALLELISM: OnceLock<AtomicUsize> = OnceLock::new();

#[inline]
fn cached_parallelism() -> usize {
    let cell = AVAILABLE_PARALLELISM.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        AtomicUsize::new(n.max(1))
    });
    // Relaxed is sufficient — the cache is immutable after the first
    // store (atomicity is structural, not ordering-driven), and we never
    // need to synchronise-through this load.
    cell.load(Ordering::Relaxed)
}

/// Moirai work-stealing backend.
///
/// # ZST
/// This is a zero-sized type — it carries no state and is
/// freely copyable. All state lives in the global Moirai runtime.
///
/// # Examples
///
/// ```
/// use coeus_core::{ComputeBackend, MoiraiBackend};
///
/// let backend = MoiraiBackend::new();
/// assert_eq!(backend.name(), "moirai");
/// assert!(backend.num_threads() >= 1);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MoiraiBackend;

impl crate::backend::traits::private::Sealed for MoiraiBackend {}

impl MoiraiBackend {
    /// Create a new handle (ZST, no allocation).
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::{ComputeBackend, MoiraiBackend};
    ///
    /// let backend = MoiraiBackend::new();
    /// assert_eq!(backend.name(), "moirai");
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for MoiraiBackend {
    type Error = crate::backend::BackendError;
    type DeviceBuffer<T: Scalar> = CpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "moirai"
    }

    /// Cached `available_parallelism()` snapshot (lock-free, syscall-free).
    #[inline]
    fn num_threads(&self) -> usize {
        cached_parallelism()
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

impl Backend for MoiraiBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let len = end.saturating_sub(start);
        if len == 0 {
            return;
        }
        // Dispatch through moirai's data-parallel surface.
        //
        // `AdaptiveWithThreshold<PARALLEL_THRESHOLD>` routes sequential for
        // small workloads (below threshold) and parallel above it.  The
        // threshold is tuned higher than moirai's default `Adaptive`
        // threshold (1024) because Coeus tensor kernels are short vectorized
        // memory-bandwidth-bound loops; the pool dispatch/join overhead
        // dominates when element counts are below a few thousand.
        //
        // Evidence: the optimizer SEQUENTIAL_THRESHOLD was measured at 65 536
        // for SGD/Adam steps (memory-only ops).  General compute kernels
        // (relu, mul, conv accumulation) have slightly more compute per
        // element, so 4 096 is a conservative crossover that avoids
        // dispatching for activation layers on small tensors while still
        // parallelising medium and large tensors.
        //
        // (The umbrella `for_each_indexed` uses BlockingTask, which targets
        // I/O-bound work, not compute — we therefore always use `SyncTask`
        // via the `AdaptiveWithThreshold` policy path.)
        const PARALLEL_THRESHOLD: usize = 4_096;
        moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<PARALLEL_THRESHOLD>, _>(
            len,
            move |i| {
                f(start + i);
            },
        );
    }
}
