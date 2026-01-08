use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::normalization::BatchNorm2d;
use nn::core::module::Module;

#[pyclass(name = "BatchNorm2d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyBatchNorm2d {
    pub inner: BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1))]
    fn new(num_features: usize, eps: Option<f64>, momentum: Option<f64>) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::default(),
                num_features,
                eps_val,
                momentum_val,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create BatchNorm2d layer: {:?}",
                    e
                ))
            })?;
        Ok(PyBatchNorm2d { inner: batchnorm })
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }

    #[getter]
    fn bias(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.bias.data().clone(),
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
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

use nn::modules::normalization::LayerNorm;

/// PyLayerNorm - Python wrapper for LayerNorm layer
#[pyclass(name = "LayerNorm", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyLayerNorm {
    pub inner: LayerNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    fn new(normalized_shape: usize, eps: Option<f64>) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let layer_norm =
            LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                vec![normalized_shape],
                eps_val,
            );
        Ok(PyLayerNorm { inner: layer_norm })
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }

    #[getter]
    fn bias(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.bias.data().clone(),
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
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyLayerNorm>()?;
    Ok(())
}
