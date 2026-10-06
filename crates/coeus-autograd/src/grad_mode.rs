//! Thread-local autograd recording mode.
//!
//! `no_grad` disables graph construction for differentiable operations while
//! preserving explicit leaf construction through [`crate::Var::new`]. This
//! matches the factory-function exception used by PyTorch: a caller can still
//! create a leaf that requires gradients inside the scope, but operations
//! executed in the scope do not allocate gradient buffers or creator nodes.

#![cfg_attr(
    all(windows, target_env = "gnu"),
    expect(
        clippy::missing_const_for_thread_local,
        reason = "the initializer is already const; rust 1.97 reports the expanded static"
    )
)]

use crate::var::Var;
use coeus_core::{ComputeBackend, Scalar};
use std::cell::Cell;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard that restores the previous autograd recording depth on drop.
///
/// Construct one via [`no_grad_guard`] to enter a no-grad scope; the recording
/// mode is restored automatically when the guard goes out of scope.
///
/// # Examples
///
/// Inside the guard's scope, differentiable ops skip graph construction, so the
/// result carries no creator node and no gradient buffer. Recording resumes on drop.
///
/// ```
/// use coeus_autograd::{Var, no_grad_guard, is_grad_enabled};
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);
/// assert!(is_grad_enabled());
///
/// {
///     let _g = no_grad_guard();
///     assert!(!is_grad_enabled());
///     let y = coeus_autograd::mul(&x, &x);
///     assert!(y.creator.is_none()); // no graph recorded
///     assert!(y.grad.is_none());    // no gradient buffer allocated
/// }
///
/// assert!(is_grad_enabled()); // restored after the guard drops
/// ```
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
///
/// Returns an RAII [`NoGradGuard`] that pushes a no-grad frame on construction
/// and pops it on drop, so the disabled-recording scope is bounded lexically.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{no_grad_guard, is_grad_enabled};
///
/// assert!(is_grad_enabled());
/// let _g = no_grad_guard();
/// assert!(!is_grad_enabled());
/// drop(_g);
/// assert!(is_grad_enabled());
/// ```
pub fn no_grad_guard() -> NoGradGuard {
    push_no_grad();
    NoGradGuard
}

/// Return whether operation graph recording is currently enabled.
///
/// Recording is enabled by default and disabled inside a [`no_grad_guard`]
/// scope (or after a manual [`push_no_grad`] without a matching [`pop_no_grad`]).
///
/// # Examples
///
/// ```
/// use coeus_autograd::{is_grad_enabled, no_grad_guard};
///
/// assert!(is_grad_enabled());
/// {
///     let _g = no_grad_guard();
///     assert!(!is_grad_enabled());
/// }
/// assert!(is_grad_enabled());
/// ```
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
