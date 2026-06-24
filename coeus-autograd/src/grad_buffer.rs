// ── GradBuffer — zero-overhead gradient accumulation cell ──
//
// Replaces `Arc<Mutex<Tensor<T,B>>>` in every backward node.
//
// # Safety Invariant
//
// `GradBuffer` exposes non-synchronized mutable access to its inner
// `Tensor`.  It is correct **only** because:
//
//   1. The backward pass (`backward_with_seed`) is always a *sequential*
//      depth-first traversal — no two nodes are executed concurrently.
//   2. No user-facing API calls `write()` outside the backward pass.
//
// If a future parallel backward path is added, this type must be replaced
// with a proper synchronization primitive (e.g. `moirai_sync::SpinLock`).

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use std::cell::UnsafeCell;

/// Zero-overhead gradient accumulation cell.
///
/// Wraps a `Tensor<T, B>` in an `UnsafeCell` so that the backward pass can
/// accumulate gradients without paying any mutex lock/unlock overhead.
///
/// See module-level documentation for the single-threaded-backward safety
/// invariant.
pub struct GradBuffer<T: Scalar, B: ComputeBackend + Default>(UnsafeCell<Tensor<T, B>>);

// SAFETY: GradBuffer is only mutated during the sequential backward pass.
// The UnsafeCell content is never accessed concurrently by Coeus autograd.
unsafe impl<T: Scalar + Send, B: ComputeBackend + Default + Send> Send for GradBuffer<T, B> {}
unsafe impl<T: Scalar + Send, B: ComputeBackend + Default + Send + Sync> Sync for GradBuffer<T, B> {}

impl<T: Scalar, B: ComputeBackend + Default> GradBuffer<T, B> {
    /// Create a new gradient buffer from an initial tensor.
    #[inline]
    pub fn new(tensor: Tensor<T, B>) -> Self {
        GradBuffer(UnsafeCell::new(tensor))
    }

    /// Get a shared reference to the inner gradient tensor.
    ///
    /// # Safety
    /// No concurrent mutable access may exist when this is called.
    #[inline]
    pub fn read(&self) -> &Tensor<T, B> {
        // SAFETY: upheld by the single-threaded-backward invariant.
        unsafe { &*self.0.get() }
    }

    /// Get a mutable reference to the inner gradient tensor.
    ///
    /// # Safety
    /// No other reference to the inner tensor may exist when this is called.
    /// This is interior mutability through `UnsafeCell` — the `mut_from_ref`
    /// lint is suppressed because this is the canonical interior-mutability
    /// pattern: the `UnsafeCell` explicitly grants permission for this.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn write(&self) -> &mut Tensor<T, B> {
        // SAFETY: upheld by the single-threaded-backward invariant.
        unsafe { &mut *self.0.get() }
    }

    /// Clone the current gradient tensor value (used for reading in `Var::grad`).
    #[inline]
    pub fn clone_tensor(&self) -> Tensor<T, B> {
        self.read().clone()
    }
}
