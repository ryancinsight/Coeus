    pub fn sort(&self, dim: usize, descending: bool) -> PyResult<(PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (v, indices) = tensor::ops::sort(t, dim, descending).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (v, indices) = tensor::ops::sort(t, dim, descending).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("sort not implemented for this storage"))
        }
    }

    pub fn topk(&self, k: usize, dim: usize, largest: bool) -> PyResult<(PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (v, indices) = tensor::ops::topk(t, k, dim, largest).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (v, indices) = tensor::ops::topk(t, k, dim, largest).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("topk not implemented for this storage"))
        }
    }

    pub fn unique(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::unique(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::unique(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("unique not implemented for this storage"))
        }
    }

    pub fn atan2(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &other.inner) {
            (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            (TensorWrapper::CpuSparseF32(a), TensorWrapper::CpuSparseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuSparseF32(res) })
            }
            (TensorWrapper::CpuSparseF64(a), TensorWrapper::CpuSparseF64(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuSparseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "atan2 not implemented for this tensor type combination (integers not supported)",
            )),
        }
    }
