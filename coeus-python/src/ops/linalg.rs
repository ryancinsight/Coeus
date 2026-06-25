// ── Vector arithmetic bindings (dot, cross) ──
//
// PyO3 wrappers for `coeus_ops::{dot, cross}`. The Python surface mirrors
// `torch.dot(input, tensor)` (returns a Python scalar) and
// `torch.cross(input, other, dim)` (returns a PyTensor of the same rank as
// the inputs). Backend mismatch between the two inputs is surfaced via
// ValueError so the Python caller can interpret a stale `MoiraiBackend` vs
// `SequentialBackend` pair cleanly at the binding boundary.

use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Inner product of two tensors.
///
/// Matches `torch.dot(input, tensor)`: both arguments are flattened and the
/// scalar `Σ aᵢ bᵢ` is returned as a Python `float`. Empty tensors, shape
/// mismatches, or dtype mismatches between the two inputs surface as
/// `ValueError` at the binding boundary rather than panicking.
#[pyfunction]
#[pyo3(name = "dot", signature = (input, tensor))]
pub fn dot(input: &PyTensor, tensor: &PyTensor, py: Python<'_>) -> PyResult<f64> {
    let a = &input.inner.tensor;
    let b = &tensor.inner.tensor;
    if a.numel() != b.numel() {
        return Err(PyValueError::new_err(format!(
            "dot: numel mismatch: input.numel()={}, tensor.numel()={}",
            a.numel(),
            b.numel()
        )));
    }
    if a.numel() == 0 {
        return Err(PyValueError::new_err(
            "dot: empty tensors have no dot product",
        ));
    }
    // Backend-aware invocation: `coeus_ops::dot` constructs its own
    // `B::default()` internally, so no caller-side backend is needed; the
    // job here is just to type-assert inference at the boundary.
    let v: f64 = py.allow_threads(|| coeus_ops::dot::<f64, MoiraiBackend>(a, b));
    Ok(v)
}

/// Per-channel 3-vector cross product along `dim`.
///
/// Matches `torch.cross(input, other, dim)`: the slice axis must have
/// exactly three elements; the output keeps the same shape (no reduction).
/// Equal-shape precondition is enforced; out-of-range or wrong-size axes
/// surface as `ValueError`.
#[pyfunction]
#[pyo3(name = "cross", signature = (input, other, dim = 0))]
pub fn cross(input: &PyTensor, other: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    let a = &input.inner.tensor;
    let b = &other.inner.tensor;
    if a.numel() == 0 {
        return Err(PyValueError::new_err(
            "cross: empty tensors have no cross product",
        ));
    }
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err(format!(
            "cross: shape mismatch: input.shape()={:?}, other.shape()={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if dim >= a.ndim() {
        return Err(PyValueError::new_err(format!(
            "cross: dim {dim} out of range for rank {}",
            a.ndim()
        )));
    }
    if a.shape()[dim] != 3 {
        return Err(PyValueError::new_err(format!(
            "cross: dim {dim} must have size 3 (got {})",
            a.shape()[dim]
        )));
    }
    let out: Tensor<f64, MoiraiBackend> =
        py.allow_threads(|| coeus_ops::cross::<f64, MoiraiBackend>(a, b, dim));
    // Wrap the auto-diff-friendly non-tracked result so subsequent ops
    // (e.g. `cross_output.sum()`) can compose through the autograd graph
    // when the caller intervenes. matches `coeus_python::ops::cumprod`.
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(out, false),
    })
}
