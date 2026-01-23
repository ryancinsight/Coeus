use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &other.inner) {
            (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
                let res = tensor::ops::matmul(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
                let res = tensor::ops::matmul(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "matmul not implemented for this tensor type combination",
            )),
        }
    }

    pub fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }



    /// Matrix multiplication alias (mm = matmul for 2D tensors)
    pub fn mm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }

    /// Computes the dot product of two 1D tensors.
    pub fn dot(&self, other: &PyTensor) -> PyResult<PyTensor> {
        // For 1D tensors, this is element-wise multiply then sum
        let mul_result = self.mul(other)?;
        mul_result.sum(None, false)
    }

    /// Computes the product of all elements.
    pub fn prod(&self) -> PyResult<PyTensor> {
        use tensor::tensor_core::Tensor;
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let data = t.as_slice();
                let prod_val = data.iter().cloned().fold(dtype::float::Float32(1.0), |acc, x| acc * x);
                let result = Tensor::from_vec(vec![prod_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let data = t.as_slice();
                let prod_val = data.iter().cloned().fold(dtype::float::Float64(1.0), |acc, x| acc * x);
                let result = Tensor::from_vec(vec![prod_val], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "prod not implemented for this tensor type",
            )),
        }
    }

    /// Computes the trace of a 2D tensor (sum of diagonal elements).
    pub fn trace(&self) -> PyResult<PyTensor> {
        use tensor::tensor_core::Tensor;
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let shape = t.shape().dims();
                if shape.len() != 2 || shape[0] != shape[1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "trace requires a square 2D tensor",
                    ));
                }
                let data = t.as_slice();
                let n = shape[0];
                let mut sum = dtype::float::Float32(0.0);
                for i in 0..n {
                    sum = sum + data[i * n + i];
                }
                let result = Tensor::from_vec(vec![sum], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(result) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let shape = t.shape().dims();
                if shape.len() != 2 || shape[0] != shape[1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "trace requires a square 2D tensor",
                    ));
                }
                let data = t.as_slice();
                let n = shape[0];
                let mut sum = dtype::float::Float64(0.0);
                for i in 0..n {
                    sum = sum + data[i * n + i];
                }
                let result = Tensor::from_vec(vec![sum], &[1]).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(result) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "trace not implemented for this tensor type",
            )),
        }
    }
}
