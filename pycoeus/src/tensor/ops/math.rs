use crate::{dispatch_tensor, dispatch_float_tensor, dispatch_tensor_mut, dispatch_ord_tensor};
use crate::tensor::class::{PyTensor, to_py_err};
use crate::tensor::wrapper::{TensorWrapper, WrapTensor};
use pyo3::prelude::*;
use dtype::int::Int64;

#[pymethods]
impl PyTensor {
    pub fn renorm(&self, p: f64, dim: i64, maxnorm: f64) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
            let rank = a.shape().ndim();
            let dim_idx = if dim < 0 { (rank as i64 + dim) as usize } else { dim as usize };

            use num_traits::NumCast;
            let p_casted = NumCast::from(p).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("renorm: failed to cast p"))?;
            let maxnorm_casted = NumCast::from(maxnorm).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("renorm: failed to cast maxnorm"))?;

            let res = tensor::ops::renorm(a, p_casted, dim_idx, maxnorm_casted).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn clamp_(&mut self, min: f64, max: f64) -> PyResult<PyTensor> {
        dispatch_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let min_casted = NumCast::from(min).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_: failed to cast min"))?;
            let max_casted = NumCast::from(max).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_: failed to cast max"))?;
            tensor::ops::clamp_(a, min_casted, max_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn clamp_min_(&mut self, min: f64) -> PyResult<PyTensor> {
        dispatch_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let min_casted = NumCast::from(min).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_min_: failed to cast min"))?;
            tensor::ops::clamp_min_(a, min_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn clamp_max_(&mut self, max: f64) -> PyResult<PyTensor> {
        dispatch_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let max_casted = NumCast::from(max).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_max_: failed to cast max"))?;
            tensor::ops::clamp_max_(a, max_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn sort(&self, dim: usize, descending: bool) -> PyResult<(PyTensor, PyTensor)> {
        dispatch_ord_tensor!(self, a => {
            let (values, indices) = tensor::ops::sort(a, dim, descending).map_err(to_py_err)?;
            let indices_i64: Vec<Int64> = indices.into_iter().map(|i| i as i64).collect();
            let indices_tensor = tensor::Tensor::from_vec(indices_i64, values.shape().dims()).map_err(to_py_err)?;
            Ok((PyTensor { inner: values.wrap() }, PyTensor { inner: indices_tensor.wrap() }))
        })
    }

    pub fn topk(&self, k: usize, dim: usize, largest: bool) -> PyResult<(PyTensor, PyTensor)> {
        dispatch_ord_tensor!(self, a => {
            let (values, indices) = tensor::ops::topk(a, k, dim, largest).map_err(to_py_err)?;
            let indices_i64: Vec<Int64> = indices.into_iter().map(|i| i as i64).collect();
            let indices_tensor = tensor::Tensor::from_vec(indices_i64, values.shape().dims()).map_err(to_py_err)?;
            Ok((PyTensor { inner: values.wrap() }, PyTensor { inner: indices_tensor.wrap() }))
        })
    }

    pub fn unique(&self, _sorted: bool, _return_inverse: bool, _return_counts: bool, _dim: Option<usize>) -> PyResult<PyTensor> {
        dispatch_ord_tensor!(self, a => {
             let res = tensor::ops::unique(a).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn square(&self) -> PyResult<PyTensor> {
        dispatch_tensor!(self, a => {
             // We need T: Float for square in current impl
             // Let's use dispatch_float_tensor or fix square to use Num
             let res = tensor::ops::square(a).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }
}
