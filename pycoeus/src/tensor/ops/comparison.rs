use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::dispatch_binary;
use pyo3::prelude::*;
use tensor::ops::comparison;
use dtype::num_traits::Zero;

#[pymethods]
impl PyTensor {
    pub fn eq(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::eq(a, b).map_err(to_py_err)?)
    }

    pub fn ne(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::ne(a, b).map_err(to_py_err)?)
    }

    pub fn gt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::gt(a, b).map_err(to_py_err)?)
    }

    pub fn ge(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::ge(a, b).map_err(to_py_err)?)
    }

    pub fn lt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::lt(a, b).map_err(to_py_err)?)
    }

    pub fn le(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => comparison::le(a, b).map_err(to_py_err)?)
    }
}

pub fn where_(_condition: &PyTensor, _input: &PyTensor, _other: &PyTensor) -> PyResult<PyTensor> {
    Err(to_py_err("where not fully implemented (requires tensor cast op)"))
}
