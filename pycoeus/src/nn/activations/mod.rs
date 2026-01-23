pub mod elu;
pub mod gelu;
pub mod glu;
pub mod relu;
pub mod shrink;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;
pub mod threshold;

pub use elu::*;
pub use gelu::*;
pub use glu::*;
pub use relu::*;
pub use shrink::*;
pub use sigmoid::*;
pub use softmax::*;
pub use tanh::*;
pub use threshold::*;

use pyo3::prelude::*;

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Activation error: {}", e))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGeLU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PyPReLU>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyLeakyReLU>()?;
    m.add_class::<PyELU>()?;
    m.add_class::<PyHardtanh>()?;
    m.add_class::<PySoftplus>()?;
    m.add_class::<PyMish>()?;
    m.add_class::<PyLogSoftmax>()?;
    m.add_class::<PySoftmax>()?;
    m.add_class::<PyReLU6>()?;
    m.add_class::<PySELU>()?;
    m.add_class::<PyHardsigmoid>()?;
    m.add_class::<PyHardswish>()?;
    m.add_class::<PyLogSigmoid>()?;
    m.add_class::<PySoftsign>()?;
    m.add_class::<PyTanhshrink>()?;
    m.add_class::<PyThreshold>()?;
    m.add_class::<PyCELU>()?;
    m.add_class::<PySoftmin>()?;
    m.add_class::<PySoftshrink>()?;
    m.add_class::<PyHardshrink>()?;
    m.add_class::<PyGLU>()?;
    m.add_class::<PyRReLU>()?;

    let dict = m.dict();
    dict.set_item("ReLU", m.getattr("ReLU")?)?;
    dict.set_item("GELU", m.getattr("GELU")?)?;
    dict.set_item("SiLU", m.getattr("SiLU")?)?;
    dict.set_item("PReLU", m.getattr("PReLU")?)?;
    dict.set_item("Tanh", m.getattr("Tanh")?)?;
    dict.set_item("Sigmoid", m.getattr("Sigmoid")?)?;
    dict.set_item("LeakyReLU", m.getattr("LeakyReLU")?)?;
    dict.set_item("ELU", m.getattr("ELU")?)?;
    dict.set_item("Hardtanh", m.getattr("Hardtanh")?)?;
    dict.set_item("Softplus", m.getattr("Softplus")?)?;
    dict.set_item("Mish", m.getattr("Mish")?)?;
    dict.set_item("LogSoftmax", m.getattr("LogSoftmax")?)?;
    dict.set_item("Softmax", m.getattr("Softmax")?)?;
    dict.set_item("ReLU6", m.getattr("ReLU6")?)?;
    dict.set_item("SELU", m.getattr("SELU")?)?;
    dict.set_item("Hardsigmoid", m.getattr("Hardsigmoid")?)?;
    dict.set_item("Hardswish", m.getattr("Hardswish")?)?;
    dict.set_item("LogSigmoid", m.getattr("LogSigmoid")?)?;
    dict.set_item("Softsign", m.getattr("Softsign")?)?;
    dict.set_item("Tanhshrink", m.getattr("Tanhshrink")?)?;
    dict.set_item("Threshold", m.getattr("Threshold")?)?;
    dict.set_item("CELU", m.getattr("CELU")?)?;
    dict.set_item("Softmin", m.getattr("Softmin")?)?;
    dict.set_item("Softshrink", m.getattr("Softshrink")?)?;
    dict.set_item("Hardshrink", m.getattr("Hardshrink")?)?;
    dict.set_item("GLU", m.getattr("GLU")?)?;
    dict.set_item("RReLU", m.getattr("RReLU")?)?;

    Ok(())
}
