use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
pub use crate::tensor::wrapper::WrapTensor;
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    pub fn tanh(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::tanh(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::tanh(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = tensor::ops::tanh(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            TensorWrapper::CpuSparseF32(t) => {
                // tanh on sparse returns sparse (tanh(0)=0)
                let res = tensor::ops::tanh(t).map_err(to_py_err)?;
                let dense = res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(dense) })
            }
            TensorWrapper::CpuSparseF64(t) => {
                let res = tensor::ops::tanh(t).map_err(to_py_err)?;
                let dense = res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(dense) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "tanh not implemented for integer tensors"
            )),
        }
    }

    pub fn sigmoid(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            TensorWrapper::CpuSparseF32(t) => {
                // Sigmoid is not structure preserving (sigmoid(0)=0.5), so it returns dense.
                let sparse_res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
                let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuSparseF64(t) => {
                let sparse_res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
                let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "sigmoid not implemented for integer tensors"
            )),
        }
    }

    pub fn relu(&self) -> PyResult<PyTensor> {
         match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::relu(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::relu(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = tensor::ops::relu(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            TensorWrapper::CpuSparseF32(t) => {
                // relu returns same storage type, convert sparse to dense
                let sparse_res = tensor::ops::relu(t).map_err(to_py_err)?;
                let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuSparseF64(t) => {
                let sparse_res = tensor::ops::relu(t).map_err(to_py_err)?;
                let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "relu not implemented for integer tensors"
            )),
        }
    }
}
