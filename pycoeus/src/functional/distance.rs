use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use crate::dispatch_float_tensor;
use crate::tensor::wrapper::WrapTensor;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, p=None, dim=None, keepdim=false))]
pub fn norm(
    input: &PyTensor,
    p: Option<f64>,
    dim: Option<usize>,
    keepdim: bool,
) -> PyResult<PyTensor> {
    dispatch_float_tensor!(input, t => {
        let dims = dim.map(|d| vec![d]);
        let dims_slice = dims.as_deref().map(|s| s as &[usize]);
        let res = ::tensor::ops::norm(t, p, dims_slice, keepdim).map_err(to_py_err)?;
        Ok(PyTensor { inner: res.wrap() })
    })
}

#[pyfunction]
#[pyo3(signature = (x1, x2, p=2.0, eps=1e-6, keepdim=false))]
pub fn pairwise_distance(
    x1: &PyTensor,
    x2: &PyTensor,
    p: f64,
    eps: f64,
    keepdim: bool,
) -> PyResult<PyTensor> {
    match (&x1.inner, &x2.inner) {
        (TensorWrapper::CpuDenseF32(t1), TensorWrapper::CpuDenseF32(t2)) => {
            let res = ::tensor::ops::pairwise_distance(t1, t2, p, eps, keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        (TensorWrapper::CpuDenseF64(t1), TensorWrapper::CpuDenseF64(t2)) => {
            let res = ::tensor::ops::pairwise_distance(t1, t2, p, eps, keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(t1), TensorWrapper::GpuDenseF32(t2)) => {
            let res = ::tensor::ops::pairwise_distance(t1, t2, p, eps, keepdim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected float tensors of same dtype",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2, dim=1, eps=1e-8))]
pub fn cosine_similarity(
    x1: &PyTensor,
    x2: &PyTensor,
    dim: usize,
    eps: f64,
) -> PyResult<PyTensor> {
    match (&x1.inner, &x2.inner) {
        (TensorWrapper::CpuDenseF32(t1), TensorWrapper::CpuDenseF32(t2)) => {
            let res = ::tensor::ops::cosine_similarity(t1, t2, dim, eps).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        (TensorWrapper::CpuDenseF64(t1), TensorWrapper::CpuDenseF64(t2)) => {
            let res = ::tensor::ops::cosine_similarity(t1, t2, dim, eps).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(t1), TensorWrapper::GpuDenseF32(t2)) => {
            let res = ::tensor::ops::cosine_similarity(t1, t2, dim, eps).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected float tensors of same dtype",
        )),
    }
}
