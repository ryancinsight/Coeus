//! The Python-exposed tensor class and its mutation guards.

use coeus_autograd::Var;
use pyo3::prelude::*;

/// Python-exposed tensor class wrapping autograd variables.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    /// Underlying Rust autograd variable holding the tensor data and gradient.
    pub inner: Var<f64>,
}

impl PyTensor {
    pub(crate) fn from_var(inner: Var<f64>) -> Self {
        Self {
            inner: crate::grad_mode::maybe_untrack_var(inner),
        }
    }

    /// Refuse an in-place rewrite of a tensor autograd is tracking.
    ///
    /// torch raises for the same case ("a leaf Variable that requires grad is
    /// being used in an in-place operation"): the recorded operation and the
    /// gradient buffer both describe the value the tensor held before the
    /// mutation, so rewriting it would leave the graph pointing at a value that
    /// no longer exists.
    ///
    /// This is deliberately stricter than torch, which allows in-place ops on a
    /// tracked *non-leaf* result: a coeus backward node saves clones of its
    /// operands at forward time, so rewriting the result would make the next
    /// backward describe the old value instead of failing. Every in-place entry
    /// point — `fill_`/`zero_`/`one_`, `__setitem__`,
    /// `__iadd__`/`__isub__`/`__imul__` — calls this first.
    ///
    /// # Errors
    /// A `ValueError` naming `op` when the tensor carries gradient state.
    pub(super) fn reject_tracked_mutation(&self, op: &str) -> PyResult<()> {
        if self.inner.grad.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{op}: cannot mutate a tensor that requires grad in place; \
                 call detach() first (or build the tensor with requires_grad = false)"
            )));
        }
        Ok(())
    }
}
