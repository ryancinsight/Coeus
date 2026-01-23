use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    pub fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.reshape(&shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.reshape(&shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            TensorWrapper::CpuDenseI64(t) => {
                let res = t.reshape(&shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.reshape(&shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "reshape not implemented for sparse tensors",
            )),
        }
    }

    pub fn view(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }

    pub fn permute(&self, dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyTensor> {
        let perm: Vec<usize> = if dims.len() == 1 {
            if let Ok(list) = dims.get_item(0).and_then(|i| i.extract::<Vec<usize>>()) {
                list
            } else {
                dims.extract::<Vec<usize>>()?
            }
        } else {
            dims.extract::<Vec<usize>>()?
        };
        
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.permute(&perm).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.permute(&perm).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            TensorWrapper::CpuDenseI64(t) => {
                let res = t.permute(&perm).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.permute(&perm).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "permute not implemented for sparse tensors",
            )),
        }
    }

    pub fn flatten(&self, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.flatten(start_dim, end_dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.flatten(start_dim, end_dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            TensorWrapper::CpuDenseI64(t) => {
                let res = t.flatten(start_dim, end_dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.flatten(start_dim, end_dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "flatten not implemented for sparse tensors",
            )),
        }
    }

    pub fn squeeze(&self, dim: Option<usize>) -> PyResult<PyTensor> {
        match dim {
            Some(d) => match &self.inner {
                TensorWrapper::CpuDenseF32(t) => {
                    let res = t.squeeze(d).map_err(to_py_err)?;
                    Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
                }
                TensorWrapper::CpuDenseF64(t) => {
                    let res = t.squeeze(d).map_err(to_py_err)?;
                    Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
                }
                TensorWrapper::CpuDenseI64(t) => {
                    let res = t.squeeze(d).map_err(to_py_err)?;
                    Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
                }
                #[cfg(feature = "gpu")]
                TensorWrapper::GpuDenseF32(t) => {
                    let res = t.squeeze(d).map_err(to_py_err)?;
                    Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "squeeze not implemented for sparse tensors",
                )),
            },
            None => {
                let mut current = self.clone();
                let dims = self.inner.shape().dims();
                let mut offset = 0;
                for (i, &size) in dims.iter().enumerate() {
                    if size == 1 {
                        let inner = match &current.inner {
                            TensorWrapper::CpuDenseF32(t) => TensorWrapper::CpuDenseF32(
                                t.squeeze(i - offset).map_err(to_py_err)?,
                            ),
                            TensorWrapper::CpuDenseF64(t) => TensorWrapper::CpuDenseF64(
                                t.squeeze(i - offset).map_err(to_py_err)?,
                            ),
                            TensorWrapper::CpuDenseI64(t) => TensorWrapper::CpuDenseI64(
                                t.squeeze(i - offset).map_err(to_py_err)?,
                            ),
                            #[cfg(feature = "gpu")]
                            TensorWrapper::GpuDenseF32(t) => TensorWrapper::GpuDenseF32(
                                t.squeeze(i - offset).map_err(to_py_err)?,
                            ),
                            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Squeeze not implemented for this tensor type")),
                        };
                        current.inner = inner;
                        offset += 1;
                    }
                }
                Ok(current)
            }
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.unsqueeze(dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.unsqueeze(dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            TensorWrapper::CpuDenseI64(t) => {
                let res = t.unsqueeze(dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.unsqueeze(dim).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "unsqueeze not implemented for sparse tensors",
            )),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.transpose(dim0, dim1).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.transpose(dim0, dim1).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            TensorWrapper::CpuDenseI64(t) => {
                let res = t.transpose(dim0, dim1).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.transpose(dim0, dim1).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "transpose not implemented for sparse tensors",
            )),
        }
    }
}
