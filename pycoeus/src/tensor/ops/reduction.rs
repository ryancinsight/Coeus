use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_ord_tensor;
use pyo3::prelude::*;

pub fn argmax(tensor: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    dispatch_ord_tensor!(tensor, t => {
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

pub fn argmin(tensor: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    dispatch_ord_tensor!(tensor, t => {
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
