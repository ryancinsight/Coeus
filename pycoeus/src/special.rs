use crate::tensor::PyTensor;
use coeus_special::bessel::Bessel;
use coeus_special::error_functions::Erf;
use coeus_special::gamma::Gamma;
use coeus_special::trig::Misc;
use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};

#[pyfunction]
pub fn erf(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.erf().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.erf failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn erfc(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.erfc().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.erfc failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn erfinv(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.erfinv().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.erfinv failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn ndtr(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.ndtr().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.ndtr failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn gamma(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.gamma().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.gamma failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn lgamma(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.lgamma().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.lgamma failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn digamma(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.digamma().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "special.digamma failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn polygamma(n: usize, input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.polygamma(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "special.polygamma failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn logit(input: &PyTensor, eps: Option<f64>) -> PyResult<PyTensor> {
    let result = input.inner.logit(eps).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.logit failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn expit(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.expit().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.expit failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn sinc(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.sinc().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("special.sinc failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn bessel_j0(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.bessel_j0().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "special.bessel_j0 failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
pub fn bessel_j1(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.bessel_j1().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "special.bessel_j1 failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}
