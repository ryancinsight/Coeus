use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::{dispatch_ord_tensor, dispatch_tensor, dispatch_float_tensor};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_ord_tensor!(self, t => {
            let res = if let Some(d) = dim {
                ::tensor::ops::argmax(t, d, keepdim).map_err(to_py_err)?
            } else {
                // Flatten and then argmax(0)
                let flattened = ::tensor::ops::flatten(t, 0, -1).map_err(to_py_err)?;
                ::tensor::ops::argmax(&flattened, 0, false).map_err(to_py_err)?
            };
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_ord_tensor!(self, t => {
            let res = if let Some(d) = dim {
                ::tensor::ops::argmin(t, d, keepdim).map_err(to_py_err)?
            } else {
                // Flatten and then argmin(0)
                let flattened = ::tensor::ops::flatten(t, 0, -1).map_err(to_py_err)?;
                ::tensor::ops::argmin(&flattened, 0, false).map_err(to_py_err)?
            };
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_tensor!(self, inner => {
            let res = ::tensor::ops::sum(inner, dim.as_ref().map(std::slice::from_ref), keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, inner => {
            let res = ::tensor::ops::mean(inner, dim.as_ref().map(std::slice::from_ref), keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, inner => {
            let res = if let Some(d) = dim {
                ::tensor::ops::max(inner, d, keepdim).map_err(to_py_err)?.wrap()
            } else {
                let flattened = ::tensor::ops::flatten(inner, 0, -1).map_err(to_py_err)?;
                ::tensor::ops::max(&flattened, 0, false).map_err(to_py_err)?.wrap()
            };
            Ok(PyTensor { inner: res })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn min(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, inner => {
            let res = if let Some(d) = dim {
                ::tensor::ops::min(inner, d, keepdim).map_err(to_py_err)?.wrap()
            } else {
                let flattened = ::tensor::ops::flatten(inner, 0, -1).map_err(to_py_err)?;
                ::tensor::ops::min(&flattened, 0, false).map_err(to_py_err)?.wrap()
            };
            Ok(PyTensor { inner: res })
        })
    }

    #[pyo3(signature = (dim=None, correction=1, keepdim=false))]
    pub fn std(&self, dim: Option<usize>, correction: usize, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, inner => {
            let res = ::tensor::ops::std(inner, dim.as_ref().map(std::slice::from_ref), keepdim, correction).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, correction=1, keepdim=false))]
    pub fn var(&self, dim: Option<usize>, correction: usize, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, inner => {
            let res = ::tensor::ops::var(inner, dim.as_ref().map(std::slice::from_ref), keepdim, correction).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn all(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_tensor!(self, inner => {
             let dims = dim.map(|d| vec![d]);
             let dims_slice = dims.as_deref().map(|s| s as &[usize]);
             
             let res = ::tensor::ops::all(inner, dims_slice, keepdim).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn any(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_tensor!(self, inner => {
             let dims = dim.map(|d| vec![d]);
             let dims_slice = dims.as_deref().map(|s| s as &[usize]);
             
             let res = ::tensor::ops::any(inner, dims_slice, keepdim).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }

    #[pyo3(signature = (p=None, dim=None, keepdim=false))]
    pub fn norm(&self, p: Option<f64>, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, t => {
            let dims = dim.map(|d| vec![d]);
            let dims_slice = dims.as_deref().map(|s| s as &[usize]);
            let res = ::tensor::ops::norm(t, p, dims_slice, keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }
}
