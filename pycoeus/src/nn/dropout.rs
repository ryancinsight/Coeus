//! Python bindings for Dropout layers.

use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::Dropout;
use nn::core::module::Module;

/// PyDropout - Python wrapper for Dropout layer
#[pyclass(name = "Dropout", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyDropout {
    pub inner: Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: Option<f64>) -> PyResult<Self> {
        let probability = p.unwrap_or(0.5);
        let dropout = Dropout::new(probability);
        Ok(PyDropout { inner: dropout })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        // Dropout has no learnable parameters
        Ok(vec![])
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDropout>()?;
    Ok(())
}
