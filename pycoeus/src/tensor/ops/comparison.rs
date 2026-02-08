use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::tensor::wrapper::WrapTensor;
use crate::{dispatch_binary, dispatch_tensor, dispatch_float_binary, dispatch_float_tensor};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
use tensor::ops::comparison;

#[pymethods]
impl PyTensor {
    pub fn eq(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::eq(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn ne(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::ne(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn gt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::gt(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn ge(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::ge(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn lt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::lt(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn le(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = comparison::le(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (other, rtol=1e-05, atol=1e-08, equal_nan=false))]
    pub fn isclose(&self, other: &PyTensor, rtol: f64, atol: f64, equal_nan: bool) -> PyResult<PyTensor> {
        dispatch_float_binary!(self, other, a, b => {
            let res = comparison::isclose(a, b, rtol, atol, equal_nan).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (other, rtol=1e-05, atol=1e-08, equal_nan=false))]
    pub fn allclose(&self, other: &PyTensor, rtol: f64, atol: f64, equal_nan: bool) -> PyResult<bool> {
        dispatch_float_binary!(self, other, a, b => {
             comparison::allclose(a, b, rtol, atol, equal_nan).map_err(to_py_err)
        })
    }

    pub fn isnan(&self) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
            let res = comparison::isnan(a).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn isinf(&self) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
            let res = comparison::isinf(a).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn isfinite(&self) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
            let res = comparison::isfinite(a).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn logical_and(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = ::tensor::ops::logical_and(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn logical_or(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = ::tensor::ops::logical_or(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn logical_xor(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = ::tensor::ops::logical_xor(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn logical_not(&self) -> PyResult<PyTensor> {
        dispatch_tensor!(self, a => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let res = ::tensor::ops::logical_not(&a_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }
}

pub fn where_(condition: &PyTensor, input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    dispatch_binary!(input, other, a, b => {
        let a_strided = a.to_strided().map_err(to_py_err)?;
        let b_strided = b.to_strided().map_err(to_py_err)?;
        dispatch_tensor!(condition, cond => {
            let cond_u8 = ::tensor::ops::cast::cast::<dtype::int::UInt8, _, _, _>(cond).map_err(to_py_err)?;
            let res = ::tensor::ops::comparison::where_cond::where_cond(&cond_u8, &a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    })
}

pub fn isnan(input: &PyTensor) -> PyResult<PyTensor> {
    input.isnan()
}

pub fn isinf(input: &PyTensor) -> PyResult<PyTensor> {
    input.isinf()
}

pub fn isfinite(input: &PyTensor) -> PyResult<PyTensor> {
    input.isfinite()
}

pub fn logical_and(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_and(other)
}

pub fn logical_or(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_or(other)
}

pub fn logical_xor(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_xor(other)
}

pub fn logical_not(input: &PyTensor) -> PyResult<PyTensor> {
    input.logical_not()
}
