//! One shared dispatch path for the tensor elementwise surface.
//!
//! Both the `PyTensor` methods (`tensor::pyimpl`) and the module-level
//! `#[pyfunction]`s (`ops::elementwise`) funnel through [`apply_unary`]
//! and [`apply_binary`]: release the GIL around the single `Var`
//! kernel call, then wrap through [`PyTensor::from_var`] so grad-mode
//! handling stays identical however the op is reached.

use crate::tensor::PyTensor;
use coeus_autograd::Var;
use pyo3::prelude::*;

/// Run one unary `Var` kernel with the GIL released.
#[inline]
pub(crate) fn apply_unary<F>(input: &PyTensor, py: Python<'_>, op: F) -> PyTensor
where
    F: FnOnce(&Var<f64>) -> Var<f64> + Send,
{
    PyTensor::from_var(py.allow_threads(|| op(&input.inner)))
}

/// Run one binary `Var` kernel with the GIL released.
///
/// Pre-clones both operands so only `Send` data crosses the GIL
/// boundary.
#[inline]
pub(crate) fn apply_binary<F>(a: &PyTensor, b: &PyTensor, py: Python<'_>, op: F) -> PyTensor
where
    F: FnOnce(&Var<f64>, &Var<f64>) -> Var<f64> + Send,
{
    let x = a.inner.clone();
    let y = b.inner.clone();
    PyTensor::from_var(py.allow_threads(move || op(&x, &y)))
}
