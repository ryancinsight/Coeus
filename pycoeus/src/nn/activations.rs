use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::activation::{ReLU, GeLU, SiLU, PReLU};
use nn::core::module::Module;

#[pyclass(name = "ReLU", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyReLU {
    pub inner: ReLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        PyReLU { inner: ReLU::new() }
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
}

#[pyclass(name = "GELU", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyGeLU {
    pub inner: GeLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyGeLU {
    #[new]
    fn new() -> Self {
        PyGeLU { inner: GeLU::new() }
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
}

#[pyclass(name = "SiLU", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PySiLU {
    pub inner: SiLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PySiLU {
    #[new]
    fn new() -> Self {
        PySiLU { inner: SiLU::new() }
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
}

#[pyclass(name = "PReLU", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyPReLU {
    pub inner: PReLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyPReLU {
    #[new]
    #[pyo3(signature = (num_parameters=1, init=0.25))]
    fn new(num_parameters: usize, init: f32) -> Self {
        PyPReLU {
            inner: PReLU::new(num_parameters, Some(Float32::new(init))),
        }
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

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }
}

#[pyclass(name = "Tanh", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyTanh {}

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> Self {
        PyTanh {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Use tensor arithmetic implementation
         let output = tensor::ops::arithmetic::tanh(&input.inner)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        Ok(PyTensor { inner: output })
    }
}

#[pyclass(name = "Sigmoid", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PySigmoid {}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        PySigmoid {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
         let output = nn::functional::ops::sigmoid(&input.inner)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        Ok(PyTensor { inner: output })
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGeLU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PyPReLU>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PySigmoid>()?;
    Ok(())
}
