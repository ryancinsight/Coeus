// ── PyTensor unary math operations ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::exp(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn erf(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::erf(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn erfc(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::erfc(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::tan(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn asin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::asin(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn acos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::acos(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn atan(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::atan(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn atanh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::atanh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn asinh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::asinh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn acosh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::acosh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn expm1(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::expm1(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log1p(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log1p(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sinh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sinh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn cosh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cosh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log2(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log10(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn exp2(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::exp2(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn selu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::selu(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::abs(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sqrt(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn recip(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::recip(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sign(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::floor(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::ceil(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn round(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::round(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn trunc(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::trunc(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sin(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cos(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn pow(&self, exp: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self::from_var(inner))
    }

    fn __pow__(&self, exp: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self::from_var(inner))
    }

    fn clamp(&self, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::clamp(&self.inner, min_val, max_val));
        Ok(Self::from_var(inner))
    }

    fn scale(&self, s: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&self.inner, s));
        Ok(Self::from_var(inner))
    }
}
