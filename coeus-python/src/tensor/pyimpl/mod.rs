// ── PyTensor struct, core helpers, and pymethods submodules ──

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
}

mod basic;
mod compare;
mod dtype;
mod grad;
mod indexing;
mod inplace;
mod math;
mod reduction;
mod shape;
