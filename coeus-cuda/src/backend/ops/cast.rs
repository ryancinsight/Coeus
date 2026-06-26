// ── Scalar reinterpretation for monomorphized CUDA kernels ──
//
// On-device kernels are compiled for concrete `f32` (NVRTC/PTX), while the
// `BackendOps<T>` trait is generic. After a `TypeId` check confirms `T == f32`,
// these helpers reinterpret a `CudaStorage<T>` as `CudaStorage<f32>` (and back)
// without copying device memory. This is the single authoritative definition;
// the op modules (conv, math, optim, attention) call into it rather than
// redefining it.

use crate::storage::CudaStorage;

/// Reinterpret a shared device buffer's element type without copying.
///
/// # Safety contract
/// The caller MUST have established `T` and `U` are the same concrete type
/// (via `TypeId`), so the `Arc<CudaBuffer<T>>` and `Arc<CudaBuffer<U>>` layouts
/// are identical. Calling with `T != U` is undefined behavior.
#[inline]
pub(super) fn cast_storage<T, U>(storage: &CudaStorage<T>) -> CudaStorage<U> {
    // SAFETY: callers gate on `TypeId::of::<T>() == TypeId::of::<f32>()` (and
    // U == f32), so the two `Arc<CudaBuffer<_>>` reprs are identical and the
    // transmute only relabels the element type of a refcounted handle.
    let buffer = unsafe {
        std::mem::transmute::<
            std::sync::Arc<hephaestus_cuda::CudaBuffer<T>>,
            std::sync::Arc<hephaestus_cuda::CudaBuffer<U>>,
        >(storage.buffer.clone())
    };
    CudaStorage { buffer }
}

/// `&mut` variant of [`cast_storage`]; identical reinterpretation semantics.
///
/// # Safety contract
/// See [`cast_storage`] — `T` and `U` must be the same concrete type.
#[inline]
pub(super) fn cast_storage_mut<T, U>(storage: &mut CudaStorage<T>) -> CudaStorage<U> {
    // SAFETY: see `cast_storage`; the `&mut` receiver does not change the
    // layout-identity argument — only the buffer handle is relabeled.
    let buffer = unsafe {
        std::mem::transmute::<
            std::sync::Arc<hephaestus_cuda::CudaBuffer<T>>,
            std::sync::Arc<hephaestus_cuda::CudaBuffer<U>>,
        >(storage.buffer.clone())
    };
    CudaStorage { buffer }
}
