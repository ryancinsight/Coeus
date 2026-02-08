use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

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
        match (&self.inner, &other.inner) {
            (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
                let res = tensor::ops::bmm(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
                let res = tensor::ops::bmm(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "bmm not implemented for this tensor type combination",
            )),
        }
    }

    /// Matrix multiplication alias (mm = matmul for 2D tensors)
    pub fn mm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }

    /// Add matrix multiplication: beta * self + alpha * (mat1 @ mat2)
    #[pyo3(signature = (mat1, mat2, *, beta=1.0, alpha=1.0))]
    pub fn addmm(&self, mat1: &PyTensor, mat2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
         match (&self.inner, &mat1.inner, &mat2.inner) {
            (TensorWrapper::CpuDenseF32(input), TensorWrapper::CpuDenseF32(m1), TensorWrapper::CpuDenseF32(m2)) => {
                let res = tensor::ops::addmm(input, m1, m2, dtype::float::Float32(beta as f32), dtype::float::Float32(alpha as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(input), TensorWrapper::CpuDenseF64(m1), TensorWrapper::CpuDenseF64(m2)) => {
                 let res = tensor::ops::addmm(input, m1, m2, dtype::float::Float64(beta), dtype::float::Float64(alpha)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addmm not implemented for this tensor type combination",
            )),
         }
    }

    /// Add matrix-vector multiplication: beta * self + alpha * (mat @ vec)
    #[pyo3(signature = (mat, vec, *, beta=1.0, alpha=1.0))]
    pub fn addmv(&self, mat: &PyTensor, vec: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
         match (&self.inner, &mat.inner, &vec.inner) {
            (TensorWrapper::CpuDenseF32(input), TensorWrapper::CpuDenseF32(m), TensorWrapper::CpuDenseF32(v)) => {
                let res = tensor::ops::addmv(input, m, v, dtype::float::Float32(beta as f32), dtype::float::Float32(alpha as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(input), TensorWrapper::CpuDenseF64(m), TensorWrapper::CpuDenseF64(v)) => {
                 let res = tensor::ops::addmv(input, m, v, dtype::float::Float64(beta), dtype::float::Float64(alpha)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addmv not implemented for this tensor type combination",
            )),
         }
    }
    
    /// Add batch matrix multiplication: beta * self + alpha * (batch1 @ batch2)
    #[pyo3(signature = (batch1, batch2, *, beta=1.0, alpha=1.0))]
    pub fn baddbmm(&self, batch1: &PyTensor, batch2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
         match (&self.inner, &batch1.inner, &batch2.inner) {
            (TensorWrapper::CpuDenseF32(input), TensorWrapper::CpuDenseF32(b1), TensorWrapper::CpuDenseF32(b2)) => {
                let res = tensor::ops::baddbmm(input, b1, b2, dtype::float::Float32(beta as f32), dtype::float::Float32(alpha as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(input), TensorWrapper::CpuDenseF64(b1), TensorWrapper::CpuDenseF64(b2)) => {
                 let res = tensor::ops::baddbmm(input, b1, b2, dtype::float::Float64(beta), dtype::float::Float64(alpha)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "baddbmm not implemented for this tensor type combination",
            )),
         }
    }

    /// Add batch matrix multiplication (summed): beta * self + alpha * (batch1 @ batch2).sum(0)
    #[pyo3(signature = (batch1, batch2, *, beta=1.0, alpha=1.0))]
    pub fn addbmm(&self, batch1: &PyTensor, batch2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
         match (&self.inner, &batch1.inner, &batch2.inner) {
            (TensorWrapper::CpuDenseF32(input), TensorWrapper::CpuDenseF32(b1), TensorWrapper::CpuDenseF32(b2)) => {
                let res = tensor::ops::addbmm(input, b1, b2, dtype::float::Float32(beta as f32), dtype::float::Float32(alpha as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(input), TensorWrapper::CpuDenseF64(b1), TensorWrapper::CpuDenseF64(b2)) => {
                 let res = tensor::ops::addbmm(input, b1, b2, dtype::float::Float64(beta), dtype::float::Float64(alpha)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addbmm not implemented for this tensor type combination",
            )),
         }
    }

    /// Matrix-vector multiplication
    pub fn mv(&self, vec: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &vec.inner) {
             (TensorWrapper::CpuDenseF32(m), TensorWrapper::CpuDenseF32(v)) => {
                 let res = tensor::ops::mv(m, v).map_err(to_py_err)?;
                  Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
             }
             (TensorWrapper::CpuDenseF64(m), TensorWrapper::CpuDenseF64(v)) => {
                 let res = tensor::ops::mv(m, v).map_err(to_py_err)?;
                  Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
             }
              _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "mv not implemented for this tensor type combination",
            )),
        }
    }

    /// Outer product of vectors: vec1 @ vec2^T
    /// If input is provided, behaves like addr: beta * input + alpha * (vec1 @ vec2)
    /// NOTE: To match PyTorch structure where `addr` is a method on input, we use this signature.
    /// self is input.
    #[pyo3(signature = (vec1, vec2, *, beta=1.0, alpha=1.0))]
    pub fn addr(&self, vec1: &PyTensor, vec2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
        match (&self.inner, &vec1.inner, &vec2.inner) {
             (TensorWrapper::CpuDenseF32(input), TensorWrapper::CpuDenseF32(v1), TensorWrapper::CpuDenseF32(v2)) => {
                 let res = tensor::ops::addr(input, v1, v2, dtype::float::Float32(beta as f32), dtype::float::Float32(alpha as f32)).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
             }
             (TensorWrapper::CpuDenseF64(input), TensorWrapper::CpuDenseF64(v1), TensorWrapper::CpuDenseF64(v2)) => {
                 let res = tensor::ops::addr(input, v1, v2, dtype::float::Float64(beta), dtype::float::Float64(alpha)).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
             }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addr not implemented for this tensor type combination",
            )),
        }
    }
    
    /// Pure outer product: vec1 @ vec2^T (no input, no beta/alpha)
    /// PyTorch has torch.outer(vec1, vec2)
    pub fn outer(&self, vec2: &PyTensor) -> PyResult<PyTensor> {
         // self is vec1
         // vec2 is vec2
          match (&self.inner, &vec2.inner) {
             (TensorWrapper::CpuDenseF32(v1), TensorWrapper::CpuDenseF32(v2)) => {
                 let res = tensor::ops::outer(v1, v2).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
             }
             (TensorWrapper::CpuDenseF64(v1), TensorWrapper::CpuDenseF64(v2)) => {
                 let res = tensor::ops::outer(v1, v2).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
             }
              _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "outer not implemented for this tensor type combination",
            )),
        }
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

    pub fn cholesky(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::cholesky(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::cholesky(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "cholesky not implemented for this tensor type",
            )),
        }
    }

    pub fn qr(&self) -> PyResult<(PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (q, r) = tensor::ops::qr(t).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(q) },
                    PyTensor { inner: TensorWrapper::CpuDenseF32(r) },
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (q, r) = tensor::ops::qr(t).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(q) },
                    PyTensor { inner: TensorWrapper::CpuDenseF64(r) },
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "qr not implemented for this tensor type",
            )),
        }
    }

    pub fn svd(&self) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (u, s, v) = tensor::ops::svd(t).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(u) },
                    PyTensor { inner: TensorWrapper::CpuDenseF32(s) },
                    PyTensor { inner: TensorWrapper::CpuDenseF32(v) },
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (u, s, v) = tensor::ops::svd(t).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(u) },
                    PyTensor { inner: TensorWrapper::CpuDenseF64(s) },
                    PyTensor { inner: TensorWrapper::CpuDenseF64(v) },
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "svd not implemented for this tensor type",
            )),
        }
    }
}
