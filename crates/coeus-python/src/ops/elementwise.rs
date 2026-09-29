use crate::{error::map_backend_error, tensor::PyTensor};
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pyfunction]
pub fn exp(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::exp(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn log(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::log(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Gauss error function (`torch.erf`); differentiable, d/dx = 2/√π·e^(−x²).
#[pyfunction]
pub fn erf(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::erf(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Complementary error function (`torch.special.erfc`); differentiable.
#[pyfunction]
pub fn erfc(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::erfc(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Natural logarithm of the absolute gamma function (`torch.special.gammaln`).
///
/// The forward path is available for non-grad tensors. Gradients require
/// `digamma`, which is not available in the current provider surface.
#[pyfunction]
pub fn gammaln(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    if input.inner.grad.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "gammaln backward requires digamma support",
        ));
    }
    let tensor = py.allow_threads(|| coeus_autograd::lgamma_forward(&input.inner));
    let tensor = tensor.map_err(map_backend_error)?;
    Var::new(tensor, false)
        .map(PyTensor::from_var)
        .map_err(map_backend_error)
}

/// Alias matching `torch.lgamma`.
#[pyfunction]
pub fn lgamma(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    gammaln(input, py)
}

#[pyfunction]
pub fn tan(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::tan(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn asin(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::asin(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn acos(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::acos(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn atan(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::atan(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn atanh(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::atanh(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn asinh(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::asinh(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn acosh(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::acosh(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn expm1(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::expm1(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn log1p(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::log1p(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn sinh(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::sinh(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn cosh(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::cosh(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn log2(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::log2(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn log10(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::log10(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn exp2(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::exp2(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn abs(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::abs(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn sqrt(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::sqrt(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn neg(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::neg(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn recip(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::recip(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn sign(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::sign(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn floor(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::floor(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn ceil(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::ceil(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn round(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::round(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn trunc(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::trunc(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::clamp(&input.inner, min_val, max_val));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_mul(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_add(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_add(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_div(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_div(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_sub(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_sub(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn pow(input: &PyTensor, exp: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::pow(&input.inner, exp));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Element-wise remainder (`torch.remainder`): `a - floor(a / b) * b`,
/// carrying the sign of the divisor. Gradient flows to `a` (identity) and
/// `b` (`-floor(a / b)`).
#[pyfunction]
pub fn remainder(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let x = a.inner.clone();
    let y = b.inner.clone();
    let inner = py.allow_threads(move || coeus_autograd::remainder(&x, &y));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Element-wise maximum (`torch.maximum`, Burn `Tensor::max_pair`). Gradient
/// routes to the larger operand; ties resolve to `a`.
#[pyfunction]
pub fn maximum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let x = a.inner.clone();
    let y = b.inner.clone();
    let inner = py.allow_threads(move || coeus_autograd::maximum(&x, &y));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Element-wise minimum (`torch.minimum`, Burn `Tensor::min_pair`). Gradient
/// routes to the smaller operand; ties resolve to `a`.
#[pyfunction]
pub fn minimum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let x = a.inner.clone();
    let y = b.inner.clone();
    let inner = py.allow_threads(move || coeus_autograd::minimum(&x, &y));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn sin(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::sin(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

#[pyfunction]
pub fn cos(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::cos(&input.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}
