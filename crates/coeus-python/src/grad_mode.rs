//! Python binding adapter for core autograd-mode state.
//!
//! The Rust autograd graph owns recording semantics. This module keeps the
//! PyO3 wrapper thin by forwarding context-manager enter/exit to `coeus-autograd`.

use coeus_autograd::Var;

pub(crate) fn push_no_grad() {
    coeus_autograd::push_no_grad();
}

pub(crate) fn pop_no_grad() {
    coeus_autograd::pop_no_grad();
}

pub(crate) fn maybe_untrack_var(var: Var<f64>) -> Var<f64> {
    if coeus_autograd::is_no_grad_enabled() {
        Var::new(var.tensor, false)
    } else {
        var
    }
}
