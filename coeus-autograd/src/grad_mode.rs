//! Thread-local autograd recording mode.
//!
//! `no_grad` disables graph construction for differentiable operations while
//! preserving explicit leaf construction through [`crate::Var::new`]. This
//! matches the factory-function exception used by PyTorch: a caller can still
//! create a leaf that requires gradients inside the scope, but operations
//! executed in the scope do not allocate gradient buffers or creator nodes.

use crate::var::Var;
use coeus_core::{ComputeBackend, Scalar};
use std::cell::Cell;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard that restores the previous autograd recording depth on drop.
#[must_use]
pub struct NoGradGuard;

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        pop_no_grad();
    }
}

/// Enter a no-grad scope on the current thread.
pub fn push_no_grad() {
    NO_GRAD_DEPTH.with(|depth| depth.set(depth.get() + 1));
}

/// Exit a no-grad scope on the current thread.
pub fn pop_no_grad() {
    NO_GRAD_DEPTH.with(|depth| {
        let current = depth.get();
        if current > 0 {
            depth.set(current - 1);
        }
    });
}

/// Create a no-grad guard for scoped Rust use.
pub fn no_grad_guard() -> NoGradGuard {
    push_no_grad();
    NoGradGuard
}

/// Return whether operation graph recording is currently enabled.
pub fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.with(|depth| depth.get() == 0)
}

/// Return whether operation graph recording is currently disabled.
pub fn is_no_grad_enabled() -> bool {
    !is_grad_enabled()
}

/// Return whether an operation should track this variable as differentiable.
#[inline]
pub(crate) fn should_track_var<T, B>(var: &Var<T, B>) -> bool
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    is_grad_enabled() && var.grad.is_some()
}
