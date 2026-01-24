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

    /// Outer product of vectors
    pub fn addr(&self, vec1: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
        // Note: addr usually performs beta * self + alpha * (vec1 @ vec2)
        // But my tensor::ops::addr implementation is JUST vec1 @ vec2.
        // Wait, PyTorch `addr` is `input + vec1 @ vec2`.
        // `torch.ger` or `torch.outer` is just outer product.
        // My implementation in `tensor` was just `vec1 @ vec2` (outer product).
        // If I want full `addr` behavior (add + outer), I need to update `tensor` or implement logic here.
        // The task said "Implement `addr` (outer product)".
        // So I will implement it as `outer`.
        // BUT PyTorch `addr` has `beta`, `alpha`.
        // If the user meant `torch.outer`, then `addr` name in `tensor` is fine.
        // If they meant `torch.addr`, then it should be `add_outer`?
        // Let's bind it as `outer` in Python if it is pure outer product.
        // But if I name it `addr`, users imply `torch.addr`.
        // User asked for "Implement `addr` (outer product)". I'll stick to that description.
        // I will bind it as `addr` but note it currently just does outer product? 
        // Or better: bind as `outer` AND `addr` (ignoring self? No that's confusing).
        // Let's check my `addr.rs` implementation: just `matmul`.
        // So it is `outer`.
        // I'll bind as `outer` primarily, and maybe `addr` but `addr` usually takes `input`.
        
        match (&self.inner, &vec1.inner, &vec2.inner) {
             (TensorWrapper::CpuDenseF32(_), TensorWrapper::CpuDenseF32(v1), TensorWrapper::CpuDenseF32(v2)) => {
                 // Ignoring self for now, just outer product
                 let res = tensor::ops::addr(v1, v2).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
             }
             (TensorWrapper::CpuDenseF64(_), TensorWrapper::CpuDenseF64(v1), TensorWrapper::CpuDenseF64(v2)) => {
                 let res = tensor::ops::addr(v1, v2).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
             }
              (TensorWrapper::CpuDenseF32(_), _, _) | (TensorWrapper::CpuDenseF64(_), _, _) => {
                  // For addr, we need to handle self if we follow torch spec.
                  // But pure outer doesn't use self.
                  // I'll implementation `outer` behavior for now.
                   Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addr (outer) not implemented for this tensor type combination",
            ))
              }
               _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "addr not implemented for this tensor type combination",
            )),
        }
    }
    
    pub fn outer(&self, vec2: &PyTensor) -> PyResult<PyTensor> {
         // self is vec1
         // vec2 is vec2
          match (&self.inner, &vec2.inner) {
             (TensorWrapper::CpuDenseF32(v1), TensorWrapper::CpuDenseF32(v2)) => {
                 let res = tensor::ops::addr(v1, v2).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
             }
             (TensorWrapper::CpuDenseF64(v1), TensorWrapper::CpuDenseF64(v2)) => {
                 let res = tensor::ops::addr(v1, v2).map_err(to_py_err)?;
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
}
