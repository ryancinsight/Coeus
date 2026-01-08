use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::linear::Linear;
use nn::core::module::Module;

#[pyclass(name = "Linear", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyLinear {
    pub inner: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    pub use_bias: bool,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create Linear layer: {:?}",
                e
            ))
        })?;
        Ok(PyLinear { inner: linear, use_bias })
    }

    fn len(&self) -> usize {
        if self.use_bias { 2 } else { 1 }
    }

    #[getter]
    fn bias(&self) -> PyResult<Option<PyTensor>> {
        if self.use_bias {
            Ok(Some(PyTensor {
                inner: self.inner.bias.data().clone(),
            }))
        } else {
            Ok(None)
        }
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
        if self.use_bias {
            // Use normal forward with bias
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
        } else {
            // Forward without bias: output = input @ weight.T
            let weight_t = self.inner.weight.data().transpose(1, 0)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Weight transpose failed: {:?}",
                        e
                    ))
                })?;
            let output = input.inner.matmul(&weight_t)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Forward pass failed: {:?}",
                        e
                    ))
                })?;
            Ok(PyTensor { inner: output })
        }
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = vec![PyTensor {
            inner: self.inner.weight.data().clone(),
        }];
        if self.use_bias {
            params.push(PyTensor {
                inner: self.inner.bias.data().clone(),
            });
        }
        Ok(params)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinear>()?;
    Ok(())
}

