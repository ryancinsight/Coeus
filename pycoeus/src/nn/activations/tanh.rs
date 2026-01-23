use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::activation::Hardtanh;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============ Tanh ============
#[pyclass(name = "Tanh", module = "coeus.nn", subclass, unsendable)]
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
        input.tanh()
    }
}

// ============ Hardtanh ============
#[pyclass(name = "Hardtanh", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyHardtanh {
    pub min_val: f64,
    pub max_val: f64,
}

#[pymethods]
impl PyHardtanh {
    #[new]
    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    fn new(min_val: f64, max_val: f64) -> Self {
        PyHardtanh { min_val, max_val }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = Hardtanh::new(
                    Float32::new(self.min_val as f32),
                    Float32::new(self.max_val as f32),
                )
                .forward(i)
                .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = Hardtanh::new(Float64::new(self.min_val), Float64::new(self.max_val))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = Hardtanh::new(
                    Float32::new(self.min_val as f32),
                    Float32::new(self.max_val as f32),
                )
                .forward(i)
                .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for Hardtanh",
            )),
        }
    }
}

// ============ Tanhshrink ============
#[pyclass(name = "Tanhshrink", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyTanhshrink;

#[pymethods]
impl PyTanhshrink {
    #[new]
    fn new() -> Self {
        PyTanhshrink
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Tanhshrink(x) = x - tanh(x)
        let tanh_result = input.tanh()?;
        input.sub(&tanh_result)
    }
}
