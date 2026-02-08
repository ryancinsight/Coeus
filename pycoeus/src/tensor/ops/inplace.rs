use crate::dispatch_tensor_mut;
use crate::tensor::class::{PyTensor, to_py_err};
use crate::tensor::wrapper::{TensorWrapper};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    pub fn zero_(&mut self) -> PyResult<PyTensor> {
        let _ = dispatch_tensor_mut!(self, inner => tensor::ops::inplace::zero_(inner).map_err(to_py_err));
        Ok(self.clone())
    }

    pub fn fill_(&mut self, value: f64) -> PyResult<PyTensor> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::inplace::fill_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::inplace::fill_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::inplace::fill_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            TensorWrapper::CpuStridedF32(inner) => tensor::ops::inplace::fill_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuStridedF64(inner) => tensor::ops::inplace::fill_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuStridedI64(inner) => tensor::ops::inplace::fill_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(inner) => tensor::ops::inplace::fill_(inner, Float32(value as f32)).map_err(to_py_err),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuStridedF32(inner) => tensor::ops::inplace::fill_(inner, Float32(value as f32)).map_err(to_py_err),
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("fill_ not implemented for this tensor type")),
        }?;
        Ok(self.clone())
    }

    pub fn add_(&mut self, value: f64) -> PyResult<PyTensor> {
        match &mut self.inner {
              TensorWrapper::CpuDenseF32(inner) => tensor::ops::inplace::add_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuDenseF64(inner) => tensor::ops::inplace::add_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuDenseI64(inner) => tensor::ops::inplace::add_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              TensorWrapper::CpuStridedF32(inner) => tensor::ops::inplace::add_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuStridedF64(inner) => tensor::ops::inplace::add_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuStridedI64(inner) => tensor::ops::inplace::add_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuDenseF32(inner) => tensor::ops::inplace::add_(inner, Float32(value as f32)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuStridedF32(inner) => tensor::ops::inplace::add_(inner, Float32(value as f32)).map_err(to_py_err),
              _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("add_ not implemented for this tensor type")),
        }?;
        Ok(self.clone())
    }

    pub fn mul_(&mut self, value: f64) -> PyResult<PyTensor> {
        match &mut self.inner {
              TensorWrapper::CpuDenseF32(inner) => tensor::ops::inplace::mul_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuDenseF64(inner) => tensor::ops::inplace::mul_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuDenseI64(inner) => tensor::ops::inplace::mul_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              TensorWrapper::CpuStridedF32(inner) => tensor::ops::inplace::mul_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuStridedF64(inner) => tensor::ops::inplace::mul_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuStridedI64(inner) => tensor::ops::inplace::mul_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuDenseF32(inner) => tensor::ops::inplace::mul_(inner, Float32(value as f32)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuStridedF32(inner) => tensor::ops::inplace::mul_(inner, Float32(value as f32)).map_err(to_py_err),
               _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("mul_ not implemented for this tensor type")),
        }?;
        Ok(self.clone())
    }

    pub fn sub_(&mut self, value: f64) -> PyResult<PyTensor> {
        match &mut self.inner {
              TensorWrapper::CpuDenseF32(inner) => tensor::ops::inplace::sub_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuDenseF64(inner) => tensor::ops::inplace::sub_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuDenseI64(inner) => tensor::ops::inplace::sub_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              TensorWrapper::CpuStridedF32(inner) => tensor::ops::inplace::sub_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuStridedF64(inner) => tensor::ops::inplace::sub_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuStridedI64(inner) => tensor::ops::inplace::sub_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuDenseF32(inner) => tensor::ops::inplace::sub_(inner, Float32(value as f32)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuStridedF32(inner) => tensor::ops::inplace::sub_(inner, Float32(value as f32)).map_err(to_py_err),
               _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("sub_ not implemented for this tensor type")),
        }?;
        Ok(self.clone())
    }

    pub fn div_(&mut self, value: f64) -> PyResult<PyTensor> {
        match &mut self.inner {
              TensorWrapper::CpuDenseF32(inner) => tensor::ops::inplace::div_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuDenseF64(inner) => tensor::ops::inplace::div_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuDenseI64(inner) => tensor::ops::inplace::div_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              TensorWrapper::CpuStridedF32(inner) => tensor::ops::inplace::div_(inner, Float32(value as f32)).map_err(to_py_err),
              TensorWrapper::CpuStridedF64(inner) => tensor::ops::inplace::div_(inner, Float64(value)).map_err(to_py_err),
              TensorWrapper::CpuStridedI64(inner) => tensor::ops::inplace::div_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuDenseF32(inner) => tensor::ops::inplace::div_(inner, Float32(value as f32)).map_err(to_py_err),
              #[cfg(feature = "gpu")]
              TensorWrapper::GpuStridedF32(inner) => tensor::ops::inplace::div_(inner, Float32(value as f32)).map_err(to_py_err),
               _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("div_ not implemented for this tensor type")),
        }?;
        Ok(self.clone())
    }
}
