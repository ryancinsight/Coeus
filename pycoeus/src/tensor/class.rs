pub use crate::tensor::wrapper::TensorWrapper;
use pyo3::prelude::*;

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

    pub fn detach(&self) -> PyTensor {
        let inner = match self.inner.clone() {
            TensorWrapper::CpuDenseF32(t) => TensorWrapper::CpuDenseF32(t.detach()),
            TensorWrapper::CpuDenseF64(t) => TensorWrapper::CpuDenseF64(t.detach()),
            TensorWrapper::CpuDenseI64(t) => TensorWrapper::CpuDenseI64(t.detach()),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => TensorWrapper::GpuDenseF32(t.detach()),
            inner => inner,
        };
        PyTensor { inner }
    }

    pub fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<PyTensor> {
        let inner = match self.inner.clone() {
            TensorWrapper::CpuDenseF32(t) => TensorWrapper::CpuDenseF32(t.requires_grad_(requires_grad)),
            TensorWrapper::CpuDenseF64(t) => TensorWrapper::CpuDenseF64(t.requires_grad_(requires_grad)),
            TensorWrapper::CpuDenseI64(t) => TensorWrapper::CpuDenseI64(t.requires_grad_(requires_grad)),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => TensorWrapper::GpuDenseF32(t.requires_grad_(requires_grad)),
            _ => { return Err(to_py_err("requires_grad_ not supported for this tensor type")); }
        };
        self.inner = inner;
        Ok(self.clone())
    }

    pub fn backward(&self) -> PyResult<()> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => { t.backward().map_err(to_py_err)?; }
            TensorWrapper::CpuDenseF64(t) => { t.backward().map_err(to_py_err)?; }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => { t.backward().map_err(to_py_err)?; }
            _ => { return Err(to_py_err("backward() not supported for this tensor type")); }
        }
        Ok(())
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
    pub fn requires_grad(&self) -> bool {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => t.requires_grad(),
            TensorWrapper::CpuDenseF64(t) => t.requires_grad(),
            TensorWrapper::CpuDenseI64(t) => t.requires_grad(),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => t.requires_grad(),
            _ => false,
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
}
