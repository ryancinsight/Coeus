use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use pyo3::prelude::*;
use tensor::tensor_core::Tensor;
use storage::StorageFromVec;

#[pymethods]
impl PyTensor {
    pub fn sum(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let result = tensor::ops::sum(t, None, false).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let result = tensor::ops::sum(t, None, false).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "sum not implemented for this tensor type",
            )),
        }
    }

    pub fn mean(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let dense = t.to_dense_generic().map_err(to_py_err)?;
                let result = tensor::ops::mean(&dense, None, false).map_err(to_py_err)?;
                // Convert back to storage type
                let storage_result = <storage::DenseStorage<dtype::float::Float32> as StorageFromVec<dtype::float::Float32>>::from_vec(result.as_slice().to_vec(), result.shape().dims()).map_err(to_py_err)?;
                let tensor_result = Tensor::from_storage(storage_result, t.backend().clone());
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(tensor_result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let dense = t.to_dense_generic().map_err(to_py_err)?;
                let result = tensor::ops::mean(&dense, None, false).map_err(to_py_err)?;
                let storage_result = <storage::DenseStorage<dtype::float::Float64> as StorageFromVec<dtype::float::Float64>>::from_vec(result.as_slice().to_vec(), result.shape().dims()).map_err(to_py_err)?;
                let tensor_result = Tensor::from_storage(storage_result, t.backend().clone());
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(tensor_result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "mean not implemented for this tensor type",
            )),
        }
    }

    pub fn max(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        // For global max (no dim), we compute max along all elements manually
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let data = t.as_slice();
                if data.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot compute max of empty tensor"));
                }
                let max_val = data.iter().cloned().fold(data[0], |a, b| if b > a { b } else { a });
                let result = Tensor::from_vec(vec![max_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let data = t.as_slice();
                if data.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot compute max of empty tensor"));
                }
                let max_val = data.iter().cloned().fold(data[0], |a, b| if b > a { b } else { a });
                let result = Tensor::from_vec(vec![max_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "max not implemented for this tensor type",
            )),
        }
    }

    pub fn min(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        // For global min (no dim), we compute min along all elements manually
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let data = t.as_slice();
                if data.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot compute min of empty tensor"));
                }
                let min_val = data.iter().cloned().fold(data[0], |a, b| if b < a { b } else { a });
                let result = Tensor::from_vec(vec![min_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let data = t.as_slice();
                if data.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot compute min of empty tensor"));
                }
                let min_val = data.iter().cloned().fold(data[0], |a, b| if b < a { b } else { a });
                let result = Tensor::from_vec(vec![min_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "min not implemented for this tensor type",
            )),
        }
    }

    pub fn argmax(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "argmax is not yet implemented in pycoeus",
        ))
    }

    pub fn argmin(&self, _dim: Option<usize>, _keepdim: bool) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "argmin is not yet implemented in pycoeus",
        ))
    }

    pub fn std(&self, _dim: Option<usize>, _keepdim: bool, unbiased: Option<bool>) -> PyResult<PyTensor> {
        let correction: usize = if unbiased.unwrap_or(true) { 1 } else { 0 };
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::std(t, None, false, correction).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::std(t, None, false, correction).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "std not implemented for this tensor type",
            )),
        }
    }

    pub fn var(&self, _dim: Option<usize>, _keepdim: bool, unbiased: Option<bool>) -> PyResult<PyTensor> {
        let correction: usize = if unbiased.unwrap_or(true) { 1 } else { 0 };
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::var(t, None, false, correction).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::var(t, None, false, correction).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "var not implemented for this tensor type",
            )),
        }
    }

    /// Returns the cumulative sum of elements along a given dimension.
    pub fn cumsum(&self, dim: usize) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::cumsum(t, dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::cumsum(t, dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "cumsum not implemented for this tensor type",
            )),
        }
    }

    /// Returns the cumulative product of elements along a given dimension.
    pub fn cumprod(&self, dim: usize) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::cumprod(t, dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::cumprod(t, dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "cumprod not implemented for this tensor type",
            )),
        }
    }
}
