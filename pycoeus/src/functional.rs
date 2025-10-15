use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};
use super::tensor::PyTensor;

/// Linear function
#[pyfunction]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    // Use the functional API from coeus_nn
    let result = coeus_nn::functional::linear(&input.inner, &weight.inner, bias.map(|b| &b.inner))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Linear failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

// TODO: Implement activation functions using functional API when available

// TODO: Implement 1D convolution when available in functional module
