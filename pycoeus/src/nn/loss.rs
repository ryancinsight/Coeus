use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::modules::loss::{BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Loss error: {}", e))
}

macro_rules! dispatch_loss_stateless {
    ($pred:expr, $target:expr, $loss_type:ident) => {{
        match (&$pred.inner, &$target.inner) {
            (TensorWrapper::CpuDenseF32(p), TensorWrapper::CpuDenseF32(t)) => {
                let res = $loss_type::new().forward(p, t).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (TensorWrapper::CpuDenseF64(p), TensorWrapper::CpuDenseF64(t)) => {
                let res = $loss_type::new().forward(p, t).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (TensorWrapper::GpuDenseF32(p), TensorWrapper::GpuDenseF32(t)) => {
                let res = $loss_type::new().forward(p, t).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input/Target tensor backend/dtype mismatch",
            )),
        }
    }};
}

#[pyclass(name = "MSELoss", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMSELoss;

#[pymethods]
impl PyMSELoss {
    #[new]
    fn new() -> Self {
        PyMSELoss
    }

    fn __call__(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input, target)
    }

    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        dispatch_loss_stateless!(input, target, MSELoss)
    }
}

#[pyclass(name = "CrossEntropyLoss", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyCrossEntropyLoss;

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    fn new() -> Self {
        PyCrossEntropyLoss
    }

    fn __call__(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input, target)
    }

    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        dispatch_loss_stateless!(input, target, CrossEntropyLoss)
    }
}

#[pyclass(name = "NLLLoss", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyNLLLoss;

#[pymethods]
impl PyNLLLoss {
    #[new]
    fn new() -> Self {
        PyNLLLoss
    }

    fn __call__(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input, target)
    }

    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        dispatch_loss_stateless!(input, target, NLLLoss)
    }
}

#[pyclass(name = "BCEWithLogitsLoss", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyBCEWithLogitsLoss;

#[pymethods]
impl PyBCEWithLogitsLoss {
    #[new]
    fn new() -> Self {
        PyBCEWithLogitsLoss
    }

    fn __call__(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input, target)
    }

    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        dispatch_loss_stateless!(input, target, BCEWithLogitsLoss)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;
    m.add_class::<PyNLLLoss>()?;
    m.add_class::<PyBCEWithLogitsLoss>()?;

    let dict = m.dict();
    dict.set_item("MSELoss", m.getattr("MSELoss")?)?;
    dict.set_item("CrossEntropyLoss", m.getattr("CrossEntropyLoss")?)?;
    dict.set_item("NLLLoss", m.getattr("NLLLoss")?)?;
    dict.set_item("BCEWithLogitsLoss", m.getattr("BCEWithLogitsLoss")?)?;

    Ok(())
}
