use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use numpy::PyUntypedArrayMethods;
use tensor::tensor_core::Tensor;

#[pymethods]
impl PyTensor {
    pub fn cpu(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(_)
            | TensorWrapper::CpuDenseF64(_)
            | TensorWrapper::CpuSparseF32(_)
            | TensorWrapper::CpuSparseF64(_)
            | TensorWrapper::CpuDenseI64(_) => Ok(self.clone()),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(inner) => {
                let cpu_tensor = inner.to_cpu_dense().map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(cpu_tensor),
                })
            }
        }
    }

    pub fn cuda(&self) -> PyResult<PyTensor> {
        #[cfg(feature = "gpu")]
        {
            match &self.inner {
                TensorWrapper::GpuDenseF32(_) => Ok(self.clone()),
                TensorWrapper::CpuDenseF32(inner) => {
                    let gpu_backend = GpuBackend::<Float32>::default();
                    let gpu_tensor = inner.to_backend(&gpu_backend).map_err(to_py_err)?;
                    Ok(PyTensor {
                        inner: TensorWrapper::GpuDenseF32(gpu_tensor),
                    })
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "CUDA transfer only implemented for F32 dense",
                )),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(crate::error::convert_error("GPU support not enabled"))
        }
    }

    pub fn to(&self, device: Option<&str>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let mut res = self.clone();
        if let Some(d) = device {
            res = match d {
                "cpu" => res.cpu()?,
                "cuda" | "gpu" => res.cuda()?,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid device: {}", d)))
            };
        }
        if let Some(dt) = dtype {
            res = match (dt, &res.inner) {
                ("float32", TensorWrapper::CpuDenseF32(_)) => res,
                ("float32", TensorWrapper::CpuDenseF64(inner)) => {
                    let data = inner.as_slice().iter().map(|f| Float32(f.get() as f32)).collect();
                    let t = Tensor::from_vec(data, inner.shape().dims()).map_err(to_py_err)?;
                    PyTensor { inner: TensorWrapper::CpuDenseF32(t) }
                }
                ("float64", TensorWrapper::CpuDenseF64(_)) => res,
                ("float64", TensorWrapper::CpuDenseF32(inner)) => {
                    let data = inner.as_slice().iter().map(|f| Float64(f.get() as f64)).collect();
                    let t = Tensor::from_vec(data, inner.shape().dims()).map_err(to_py_err)?;
                    PyTensor { inner: TensorWrapper::CpuDenseF64(t) }
                }
                _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!("Dtype conversion to {} not implemented", dt)))
            };
        }
        Ok(res)
    }

    pub fn to_sparse(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let mut data = Vec::new();
                let mut row_indices = Vec::new();
                let mut col_indices = Vec::new();
                let dims = t.shape().dims();
                let slice = t.as_slice();
                for r in 0..dims[0] {
                    for c in 0..dims[1] {
                        let idx = r * dims[1] + c;
                        let val = slice[idx];
                        if val.get() != 0.0 {
                            data.push(val);
                            row_indices.push(r);
                            col_indices.push(c);
                        }
                    }
                }
                let coo = storage::CooStorage::new(data, row_indices, col_indices, dims).map_err(to_py_err)?;
                let csr = coo.to_csr().map_err(to_py_err)?;
                let sparse_tensor = Tensor::from_storage(csr, t.backend().clone());
                Ok(PyTensor { inner: TensorWrapper::CpuSparseF32(sparse_tensor) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("to_sparse only implemented for CPU dense F32"))
        }
    }

    pub fn to_dense(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuDenseI64(_) => Ok(self.clone()),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => Ok(self.clone()),
            TensorWrapper::CpuSparseF32(t) => {
                let dense_t = t.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(dense_t) })
            }
            TensorWrapper::CpuSparseF64(t) => {
                let dense_t = t.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(dense_t) })
            }
        }
    }

    pub fn item(&self) -> PyResult<f64> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                if inner.shape().size() != 1 { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("item() needs single-element tensor")); }
                Ok(inner.as_slice()[0].get() as f64)
            }
            TensorWrapper::CpuDenseF64(inner) => {
                if inner.shape().size() != 1 { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("item() needs single-element tensor")); }
                Ok(inner.as_slice()[0].get())
            }
            TensorWrapper::CpuDenseI64(inner) => {
                if inner.shape().size() != 1 { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("item() needs single-element tensor")); }
                Ok(inner.as_slice()[0].get() as f64)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("item() not implemented for this type"))
        }
    }

    pub fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.__array__(py, None, None)
    }

    pub fn __array__(&self, py: Python, _dtype: Option<Py<PyAny>>, _context: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        use numpy::{PyArray1, PyArrayMethods};
        match &self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                let data: Vec<f64> = inner.as_slice().iter().map(|f| f.get() as f64).collect();
                let array = PyArray1::from_vec(py, data);
                let shaped_array = array.reshape(inner.shape().dims()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
                Ok(shaped_array.into())
            }
            TensorWrapper::CpuDenseF64(inner) => {
                let data: Vec<f64> = inner.as_slice().iter().map(|f| f.get()).collect();
                let array = PyArray1::from_vec(py, data);
                let shaped_array = array.reshape(inner.shape().dims()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
                Ok(shaped_array.into())
            }
            TensorWrapper::CpuDenseI64(inner) => {
                let data: Vec<i64> = inner.as_slice().iter().map(|f| f.get()).collect();
                let array = PyArray1::from_vec(py, data);
                let shaped_array = array.reshape(inner.shape().dims()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
                Ok(shaped_array.into())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("__array__ only for dense CPU"))
        }
    }
}
