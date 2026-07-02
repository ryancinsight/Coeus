use crate::tensor::PyTensor;
use pyo3::prelude::*;

#[pyfunction]
pub fn exp(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::exp(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn log(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log(&input.inner));
    PyTensor::from_var(inner)
}

/// Gauss error function (`torch.erf`); differentiable, d/dx = 2/√π·e^(−x²).
#[pyfunction]
pub fn erf(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::erf(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn abs(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::abs(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn sqrt(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sqrt(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn neg(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::neg(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn recip(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::recip(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn sign(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sign(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn floor(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::floor(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn ceil(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::ceil(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn round(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::round(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn trunc(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::trunc(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::clamp(&input.inner, min_val, max_val));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn pow(input: &PyTensor, exp: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::pow(&input.inner, exp));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn sin(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sin(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn cos(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::cos(&input.inner));
    PyTensor::from_var(inner)
}
