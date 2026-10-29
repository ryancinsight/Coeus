//! Arithmetic and reverse-arithmetic operators, with scalar broadcasting.

use super::dispatch::BinOp;
use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    //
    // All five binary operators (`+`, `-`, `*`, `/`, `@`) accept either a
    // `Tensor` (elementwise) or a Python `float` scalar (broadcast via
    // `coeus_autograd::scalar_*`). The runtime discrimination is at the
    // PyO3 boundary (single arity per Python method), so the operator
    // wrappers take `&Bound<'_, PyAny>` and route to the right kernel.
    //
    // The reflected-side (e.g. `float + Tensor`) wrappers (`__radd__`,
    // `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`) receive only
    // scalars because Python's binary-op dispatch sends a `Tensor` to
    // both sides' reflected methods when the LHS-typed op fails; those
    // operators only enter when the LHS has no `+`/`*` method or
    // returned NotImplemented, so the reflected overloads are guaranteed
    // to be invoked with a non-Tensor LHS.

    fn __add__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__add__", other, BinOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__sub__", other, BinOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__mul__", other, BinOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__truediv__", other, BinOp::Div)
    }

    fn __matmul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::matmul(&self.inner, &other.inner));
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::neg)
    }

    fn __abs__(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::abs)
    }

    fn __rmul__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::scalar_mul(x, scalar))
    }

    fn __radd__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::scalar_add(x, scalar))
    }

    fn __rsub__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        // `scalar - self`: negate, then add the scalar.
        let inner = py.allow_threads(|| {
            let neg = coeus_autograd::neg(&self.inner)?;
            coeus_autograd::scalar_add(&neg, scalar)
        });
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    fn __rtruediv__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        // `scalar / self`: `scalar * reciprocal(self)`.
        let inner = py.allow_threads(|| {
            let recip = coeus_autograd::recip(&self.inner)?;
            coeus_autograd::scalar_mul(&recip, scalar)
        });
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    fn __rpow__(&self, base: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        // `base ** self`: exp(self * ln(base)).
        let inner = py.allow_threads(|| {
            let scaled = coeus_autograd::scalar_mul(&self.inner, base.ln())?;
            coeus_autograd::exp(&scaled)
        });
        inner.map(Self::from_var).map_err(map_backend_error)
    }
}
