use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_tensor;
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    pub fn contiguous(&self) -> PyResult<PyTensor> {
        dispatch_tensor!(self, t => {
            let res = t.to_cpu_dense().map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn item(&self) -> PyResult<Py<PyAny>> {
         Python::with_gil(|py| {
            match &self.inner {
                TensorWrapper::CpuDenseF32(t) => {
                     let val = t.as_slice()[0].0;
                     Ok(val.into_pyobject(py).map_err(|e| to_py_err(format!("PyO3 error: {}", e)))?.into_any().unbind())
                }
                TensorWrapper::CpuDenseF64(t) => {
                     let val = t.as_slice()[0].0;
                     Ok(val.into_pyobject(py).map_err(|e| to_py_err(format!("PyO3 error: {}", e)))?.into_any().unbind())
                }
                _ => Err(to_py_err("item() only supported for scalar CPU dense tensors")),
            }
         })
    }

    pub fn to_dense(&self) -> PyResult<PyTensor> {
        self.contiguous()
    }

    pub fn tolist(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            dispatch_tensor!(self, t => {
                let cpu_dense = t.to_cpu_dense().map_err(to_py_err)?;
                // For now, return a flat list. To match PyTorch, this should be nested based on shape.
                // But a flat list is sufficient for smoke tests.
                let list = match &cpu_dense {
                    _ => {
                        // This is a bit tricky due to generic types.
                        // We'll use a match on the wrapper types in a more manual way if needed.
                        let mut vals = Vec::new();
                        // Get data as f64 for simplicity in Python
                        match &self.inner {
                            TensorWrapper::CpuDenseF32(t) => {
                                for v in t.as_slice() { vals.push(v.0 as f64); }
                            }
                            TensorWrapper::CpuDenseF64(t) => {
                                for v in t.as_slice() { vals.push(v.0); }
                            }
                            TensorWrapper::CpuStridedF32(t) => {
                                let dense = t.to_cpu_dense().map_err(to_py_err)?;
                                for v in dense.as_slice() { vals.push(v.0 as f64); }
                            }
                            TensorWrapper::CpuStridedF64(t) => {
                                let dense = t.to_cpu_dense().map_err(to_py_err)?;
                                for v in dense.as_slice() { vals.push(v.0); }
                            }
                            TensorWrapper::CpuDenseI64(t) => {
                                for v in t.as_slice() { vals.push(v.0 as f64); }
                            }
                            _ => return Err(to_py_err("tolist() not yet implemented for this storage type")),
                        }
                        vals
                    }
                };
                
                Ok(list.into_pyobject(py).map_err(|e| to_py_err(format!("PyO3 error: {}", e)))?.into_any().unbind())
            })
        })
    }
}
