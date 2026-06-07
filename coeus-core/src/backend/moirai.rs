// ── Moirai backend ──
// Zero-sized type dispatching work to the Moirai work-stealing engine.

use crate::backend::{Backend, ComputeBackend};
use crate::storage::{Storage, CpuStorage};
use crate::dtype::Scalar;

/// Moirai work-stealing backend.
///
/// # ZST
/// This is a zero-sized type — it carries no state and is
/// freely copyable. All state lives in the global Moirai runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoiraiBackend;

impl crate::backend::traits::private::Sealed for MoiraiBackend {}

impl MoiraiBackend {
    /// Create a new handle (ZST, no allocation).
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for MoiraiBackend {
    type DeviceBuffer<T: Scalar> = CpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "moirai"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
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
        // Dispatch to Moirai global runtime
        moirai::global()
            .for_each_indexed(len, move |i| {
                f(start + i);
            })
            .expect("Moirai parallel_for dispatch failed");
    }
}
