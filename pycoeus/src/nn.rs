use pyo3::prelude::*;
use pyo3::pyclass;
use coeus_nn::{Linear, ReLU, Module};
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

// Forward declaration for PyTensor
use super::tensor::PyTensor;

/// Linear layer
#[pyclass(name = "Linear", module = "_coeus")]
pub struct PyLinear {
    pub inner: Linear<CpuBackend, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features))]
    fn new(in_features: usize, out_features: usize) -> PyResult<Self> {
        let linear = Linear::new(in_features, out_features).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create Linear layer: {:?}", e))
        })?;
        Ok(PyLinear { inner: linear })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend, DenseStorage<Float32>, Float32>::forward(&self.inner, &input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward pass failed: {:?}", e))
        })?;
        Ok(PyTensor { inner: output })
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features
    }
}

/// ReLU activation
#[pyclass(name = "ReLU", module = "_coeus")]
pub struct PyReLU {
    pub inner: ReLU,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyReLU { inner: ReLU })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend, DenseStorage<Float32>, Float32>::forward(&self.inner, &input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward pass failed: {:?}", e))
        })?;
        Ok(PyTensor { inner: output })
    }
}
