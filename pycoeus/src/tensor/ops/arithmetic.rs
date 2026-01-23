use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::dispatch_binary;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    // Binary Arithmetic Ops
    pub fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => a + b)
    }
    pub fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => a - b)
    }
    pub fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => a * b)
    }
    pub fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => a / b)
    }
    
    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> { self.__add__(other) }
    pub fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> { self.__sub__(other) }
    pub fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> { self.__mul__(other) }
    pub fn div(&self, other: &PyTensor) -> PyResult<PyTensor> { self.__truediv__(other) }
    
    pub fn pow(&self, exponent: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &exponent.inner) {
            (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
                let res = tensor::ops::pow(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
                let res = tensor::ops::pow(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
                let res = tensor::ops::pow(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "pow not implemented for these types",
            )),
        }
    }
}
