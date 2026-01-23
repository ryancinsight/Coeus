use super::{to_py_err, PyHardsigmoid};
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::activation::{GeLU, Mish, SiLU};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

macro_rules! dispatch_stateless {
    ($input:expr, $inner_type:ident) => {
        match &$input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = $inner_type::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = $inner_type::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = $inner_type::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Sparse tensors not supported for this activation",
            )),
        }
    };
}

// ============ GELU ============
#[pyclass(name = "GELU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyGeLU;

#[pymethods]
impl PyGeLU {
    #[new]
    fn new() -> Self {
        PyGeLU
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        dispatch_stateless!(input, GeLU)
    }
}

// ============ SiLU ============
#[pyclass(name = "SiLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySiLU;

#[pymethods]
impl PySiLU {
    #[new]
    fn new() -> Self {
        PySiLU
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        dispatch_stateless!(input, SiLU)
    }
}

// ============ Mish ============
#[pyclass(name = "Mish", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMish;

#[pymethods]
impl PyMish {
    #[new]
    fn new() -> Self {
        PyMish
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        dispatch_stateless!(input, Mish)
    }
}

// ============ Hardswish ============
#[pyclass(name = "Hardswish", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyHardswish;

#[pymethods]
impl PyHardswish {
    #[new]
    fn new() -> Self {
        PyHardswish
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Hardswish(x) = x * hardsigmoid(x) = x * clamp((x + 3) / 6, 0, 1)
        let hardsigmoid = PyHardsigmoid::new();
        let hs_result = hardsigmoid.forward(input)?;
        input.mul(&hs_result)
    }
}
