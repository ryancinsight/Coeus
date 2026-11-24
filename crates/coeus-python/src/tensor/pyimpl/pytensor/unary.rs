//! Parameterized unary math that the shared elementwise table does not cover.

use super::PyTensor;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn selu(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::selu)
    }

    fn pow(&self, exp: f64, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::pow(x, exp))
    }

    fn __pow__(&self, exp: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::pow(x, exp))
    }

    fn clamp(&self, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::clamp(x, min_val, max_val))
    }

    fn scale(&self, s: f64, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::scalar_mul(x, s))
    }
}
