pub use crate::tensor::wrapper::TensorWrapper;
use pyo3::prelude::*;
use crate::tensor::wrapper::WrapTensor;
use crate::{dispatch_tensor, dispatch_binary, dispatch_dense_tensor, dispatch_float_dense_tensor};
use pyo3::types::PyAnyMethods; // For into_pyobject? or just pyo3 prelude covers it.

#[pyclass(name = "Tensor", subclass)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    pub inner: TensorWrapper,
}

#[derive(Clone, Debug)]
#[pyclass(name = "Device")]
pub enum Device {
    CPU,
    GPU,
}

pub fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

#[pymethods]
impl PyTensor {
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    pub fn ndim(&self) -> usize {
        self.inner.shape().dims().len()
    }

    pub fn clone(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.clone(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype={:?})", self.shape(), self.dtype_name())
    }


    pub fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::getitem(self, index)
    }

    #[getter]
    pub fn device(&self) -> Device {
        match &self.inner {
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) | TensorWrapper::GpuStridedF32(_) |
            TensorWrapper::GpuDenseF64(_) | TensorWrapper::GpuStridedF64(_) => Device::GPU,
            _ => Device::CPU,
        }
    }

    #[getter]
    pub fn dtype_name(&self) -> String {
        match &self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuStridedF32(_) => "float32".to_string(),
            TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuStridedF64(_) => "float64".to_string(),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) | TensorWrapper::GpuStridedF32(_) => "float32".to_string(),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF64(_) | TensorWrapper::GpuStridedF64(_) => "float64".to_string(),
            TensorWrapper::CpuDenseI64(_) | TensorWrapper::CpuStridedI64(_) => "int64".to_string(),
            TensorWrapper::CpuDenseC32(_) | TensorWrapper::CpuStridedC32(_) => "complex32".to_string(),
            _ => "unknown".to_string(),
        }
    }

    #[getter]
    pub fn grad(&self) -> PyResult<Option<PyTensor>> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                if let Ok(grad_tensor) = inner.grad() {
                     Ok(Some(PyTensor { inner: TensorWrapper::CpuDenseF32(grad_tensor) }))
                } else {
                     Ok(None)
                }
            }
            TensorWrapper::CpuDenseF64(inner) => {
                if let Ok(grad_tensor) = inner.grad() {
                     Ok(Some(PyTensor { inner: TensorWrapper::CpuDenseF64(grad_tensor) }))
                } else {
                     Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    pub fn size(&self, dim: Option<usize>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let dims = self.inner.shape().dims();
            if let Some(d) = dim {
                if d >= dims.len() {
                    return Err(to_py_err(format!("Dimension out of range: {}", d)));
                }
                Ok(dims[d].into_pyobject(py).map_err(|e| to_py_err(format!("PyO3 error: {}", e)))?.into_any().unbind())
            } else {
                Ok(dims.to_vec().into_pyobject(py).map_err(|e| to_py_err(format!("PyO3 error: {}", e)))?.into_any().unbind())
            }
        })
    }

    pub fn squeeze(&self, dim: Option<usize>) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        dispatch_dense_tensor!(dense, t => {
            let res = if let Some(d) = dim {
                ::tensor::ops::squeeze(t, d).map_err(to_py_err)?.wrap()
            } else {
                t.clone().wrap()
            };
            Ok(PyTensor { inner: res })
        })
    }

    pub fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        dispatch_dense_tensor!(dense, t => {
            let res = ::tensor::ops::unsqueeze(t, dim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        dispatch_tensor!(self, t => {
            let res = ::tensor::ops::reshape(t, &shape).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn view(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        dispatch_tensor!(self, t => {
            let res = ::tensor::ops::reshape(t, &shape).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn flatten(&self, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
        dispatch_tensor!(self, t => {
            let res = ::tensor::ops::flatten(t, start_dim, end_dim).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        dispatch_tensor!(self, t => {
            let res = ::tensor::ops::transpose(t, dim0, dim1).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn permute(&self, dims: Vec<usize>) -> PyResult<PyTensor> {
        let res = self.inner.permute(&dims).map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let a = self.contiguous()?;
        let b = other.contiguous()?;
        dispatch_binary!(a, b, a_inner, b_inner => {
            let res = ::tensor::ops::matmul(a_inner, b_inner).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }

    pub fn addmm(&self, mat1: &PyTensor, mat2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
        let m12 = mat1.matmul(mat2)?;
        let scaled_m12 = m12.mul_scalar(alpha)?;
        let scaled_self = self.mul_scalar(beta)?;
        scaled_self.add(&scaled_m12)
    }

    pub fn mv(&self, vec: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(vec)
    }

    pub fn addr(&self, vec1: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
        let m = vec1.unsqueeze(1)?.matmul(&vec2.unsqueeze(0)?)?;
        self.add(&m)
    }

    pub fn outer(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.unsqueeze(1)?.matmul(&other.unsqueeze(0)?)
    }

    pub fn abs(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseC32(_) | TensorWrapper::CpuStridedC32(_) => {
                Err(to_py_err("Abs for complex not supported yet"))
            }
            _ => {
                let dense = self.contiguous()?;
                match &dense.inner {
                     TensorWrapper::CpuDenseF32(t) => {
                         let res = ::tensor::ops::abs(t).map_err(to_py_err)?;
                         Ok(PyTensor { inner: res.wrap() })
                     }
                     TensorWrapper::CpuDenseF64(t) => {
                         let res = ::tensor::ops::abs(t).map_err(to_py_err)?;
                         Ok(PyTensor { inner: res.wrap() })
                     }
                     TensorWrapper::CpuDenseI64(t) => {
                         let res = ::tensor::ops::abs(t).map_err(to_py_err)?;
                         Ok(PyTensor { inner: res.wrap() })
                     }
                     _ => Err(to_py_err("Abs requires dense storage")),
                }
            }
        }
    }

    pub fn neg(&self) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        dispatch_dense_tensor!(dense, t => {
            let res = ::tensor::ops::neg(t).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn sigmoid(&self) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        dispatch_float_dense_tensor!(dense, t => {
            let res = ::tensor::ops::sigmoid(t).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn tanh(&self) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        dispatch_float_dense_tensor!(dense, t => {
            let res = ::tensor::ops::tanh(t).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn clamp(&self, min: f64, max: f64) -> PyResult<PyTensor> {
        let dense = self.contiguous()?;
        match &dense.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let min_t = dtype::float::Float32(min as f32);
                let max_t = dtype::float::Float32(max as f32);
                let res = t.clamp(min_t, max_t).map_err(to_py_err)?;
                Ok(PyTensor { inner: res.wrap() })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let min_t = dtype::float::Float64(min);
                let max_t = dtype::float::Float64(max);
                let res = t.clamp(min_t, max_t).map_err(to_py_err)?;
                Ok(PyTensor { inner: res.wrap() })
            }
            _ => Err(to_py_err("Clamp not implemented for this storage")),
        }
    }

    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        crate::tensor::ops::reduction::argmax(self, dim, keepdim)
    }

    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        crate::tensor::ops::reduction::argmin(self, dim, keepdim)
    }

    pub fn gather(&self, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::gather(self, dim, index)
    }

    pub fn take(&self, indices: &PyTensor) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::take(self, indices)
    }

    pub fn put(&self, indices: &PyTensor, values: &PyTensor, accumulate: Option<bool>) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::put(self, indices, values, accumulate.unwrap_or(false))
    }

    pub fn masked_select(&self, mask: &PyTensor) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::masked_select(self, mask)
    }

    pub fn masked_fill(&self, mask: &PyTensor, value: f64) -> PyResult<PyTensor> {
        crate::tensor::ops::indexing::masked_fill(self, mask, value)
    }

    #[pyo3(name = "where")]
    pub fn where_(&self, condition: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
         crate::tensor::ops::comparison::where_(condition, self, other)
    }
}
