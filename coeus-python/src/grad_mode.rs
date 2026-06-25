//! Python binding autograd-mode state.
//!
//! The Rust autograd graph remains the source of tensor semantics.  This module
//! owns only Python context-manager state and strips graph edges from values
//! returned through the PyO3 boundary while `no_grad` is active.

use coeus_autograd::Var;
use std::cell::Cell;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn push_no_grad() {
    NO_GRAD_DEPTH.with(|depth| depth.set(depth.get() + 1));
}

pub(crate) fn pop_no_grad() {
    NO_GRAD_DEPTH.with(|depth| {
        let current = depth.get();
        if current > 0 {
            depth.set(current - 1);
        }
    });
}

fn no_grad_enabled() -> bool {
    NO_GRAD_DEPTH.with(|depth| depth.get() > 0)
}

pub(crate) fn maybe_untrack_var(var: Var<f64>) -> Var<f64> {
    if no_grad_enabled() {
        Var::new(var.tensor, false)
    } else {
        var
    }
}
