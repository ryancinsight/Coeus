use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_binary;
use pyo3::prelude::*;
use dtype::float::{Float32, Float64};

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let res = a.clone() + b.clone();
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let res = a.clone() - b.clone();
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let res = a.clone() * b.clone();
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let res = a.clone() / b.clone();
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn add_scalar(&self, value: f64) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => Ok(PyTensor { inner: (t.add_scalar(Float32(value as f32)).map_err(to_py_err)?).wrap() }),
            TensorWrapper::CpuDenseF64(t) => Ok(PyTensor { inner: (t.add_scalar(Float64(value)).map_err(to_py_err)?).wrap() }),
            _ => Err(to_py_err("Add scalar not implemented for this storage")),
        }
    }

    pub fn add_scalar_f64(&self, value: f64) -> PyResult<PyTensor> {
        self.add_scalar(value)
    }

    pub fn sub_scalar(&self, value: f64) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => Ok(PyTensor { inner: (t.sub_scalar(Float32(value as f32)).map_err(to_py_err)?).wrap() }),
            TensorWrapper::CpuDenseF64(t) => Ok(PyTensor { inner: (t.sub_scalar(Float64(value)).map_err(to_py_err)?).wrap() }),
            _ => Err(to_py_err("Sub scalar not implemented for this storage")),
        }
    }

    pub fn mul_scalar(&self, value: f64) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => Ok(PyTensor { inner: (t.mul_scalar(Float32(value as f32)).map_err(to_py_err)?).wrap() }),
            TensorWrapper::CpuDenseF64(t) => Ok(PyTensor { inner: (t.mul_scalar(Float64(value)).map_err(to_py_err)?).wrap() }),
            _ => Err(to_py_err("Mul scalar not implemented for this storage")),
        }
    }

    pub fn div_scalar(&self, value: f64) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => Ok(PyTensor { inner: (t.div_scalar(Float32(value as f32)).map_err(to_py_err)?).wrap() }),
            TensorWrapper::CpuDenseF64(t) => Ok(PyTensor { inner: (t.div_scalar(Float64(value)).map_err(to_py_err)?).wrap() }),
            _ => Err(to_py_err("Div scalar not implemented for this storage")),
        }
    }
}
