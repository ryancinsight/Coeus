use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_tensor;
use pyo3::prelude::*;

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
}
