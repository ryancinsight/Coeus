//! Elementwise unary math.

use super::PyTensor;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::exp)
    }

    fn erf(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::erf)
    }

    fn erfc(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::erfc)
    }

    fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::tan)
    }

    fn asin(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::asin)
    }

    fn acos(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::acos)
    }

    fn atan(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::atan)
    }

    fn atanh(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::atanh)
    }

    fn asinh(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::asinh)
    }

    fn acosh(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::acosh)
    }

    fn expm1(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::expm1)
    }

    fn log1p(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::log1p)
    }

    fn sinh(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::sinh)
    }

    fn cosh(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::cosh)
    }

    fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::log2)
    }

    fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::log10)
    }

    fn exp2(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::exp2)
    }

    fn selu(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::selu)
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::log)
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::abs)
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::sqrt)
    }

    fn recip(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::recip)
    }

    fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::sign)
    }

    fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::floor)
    }

    fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::ceil)
    }

    fn round(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::round)
    }

    fn trunc(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::trunc)
    }

    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::sin)
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::cos)
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
