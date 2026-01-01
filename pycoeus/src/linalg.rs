use crate::tensor::PyTensor;
use coeus_linalg::{Cholesky, Det, Inverse, Norm, Solve, QR, SVD};
use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};
use tensor::Float32;

/// Computes the inverse of a square matrix.
#[pyfunction]
pub fn inv(input: &PyTensor) -> PyResult<PyTensor> {
    // Delegate to linalg crate
    // Note: We need to bridge PyTensor <-> Tensor
    let result = input.inner.inv().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.inv failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

/// Computes the Frobenius norm of a tensor.
#[pyfunction]
#[pyo3(signature = (input))]
pub fn norm(input: &PyTensor) -> PyResult<f64> {
    // Frobenius norm (default)
    let result: Float32 = input.inner.norm().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.norm failed: {:?}", e))
    })?;
    Ok(f64::from(result))
}

/// Computes the vector norm of a tensor.
#[pyfunction]
#[pyo3(signature = (input, ord))]
pub fn vector_norm(input: &PyTensor, ord: f64) -> PyResult<f64> {
    // p-norm
    let result: Float32 = input.inner.norm_p(Float32(ord as f32)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "linalg.vector_norm failed: {:?}",
            e
        ))
    })?;
    Ok(f64::from(result))
}

/// Computes the determinant of a square matrix.
#[pyfunction]
pub fn det(input: &PyTensor) -> PyResult<f64> {
    let result: Float32 = input.inner.det().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.det failed: {:?}", e))
    })?;
    Ok(f64::from(result))
}

/// Solves a linear system of equations Ax = B.
#[pyfunction]
pub fn solve(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.solve(&other.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.solve failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

/// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
#[pyfunction]
pub fn cholesky(input: &PyTensor) -> PyResult<PyTensor> {
    let result = input.inner.cholesky().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "linalg.cholesky failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Computes the QR decomposition of a matrix.
/// Returns (Q, R).
#[pyfunction]
pub fn qr(input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    let result = input.inner.qr().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.qr failed: {:?}", e))
    })?;
    Ok((PyTensor { inner: result.q }, PyTensor { inner: result.r }))
}

/// Computes the Singular Value Decomposition (SVD) of a matrix.
/// Returns (U, S, Vh).
#[pyfunction]
#[pyo3(signature = (input, full_matrices=false))]
pub fn svd(input: &PyTensor, full_matrices: bool) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let result = input.inner.svd(full_matrices).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("linalg.svd failed: {:?}", e))
    })?;
    Ok((
        PyTensor { inner: result.u },
        PyTensor { inner: result.s },
        PyTensor { inner: result.vh },
    ))
}
