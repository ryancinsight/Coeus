//! Python bindings for Embedding layer.

use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::embedding::Embedding;
use nn::core::module::Module;

/// PyEmbedding - Python wrapper for Embedding layer
#[pyclass(name = "Embedding", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyEmbedding {
    pub inner: Embedding<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None))]
    fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> PyResult<Self> {
        let embedding = Embedding::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_embeddings,
            embedding_dim,
            padding_idx,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create Embedding layer: {:?}",
                e
            ))
        })?;
        Ok(PyEmbedding { inner: embedding })
    }

    #[getter]
    fn num_embeddings(&self) -> usize {
        self.inner.num_embeddings
    }

    #[getter]
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
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
        let params = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbedding>()?;
    Ok(())
}
