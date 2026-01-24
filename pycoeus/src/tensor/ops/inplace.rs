use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::dispatch_tensor_mut;
use pyo3::prelude::*;
use dtype::float::{Float32, Float64};

#[pymethods]
impl PyTensor {
    pub fn zero_(&mut self) -> PyResult<()> {
        dispatch_tensor_mut!(self, inner => tensor::ops::zero_(inner).map_err(to_py_err));
        Ok(())
    }

    pub fn fill_(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::fill_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::fill_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::fill_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "GPU fill_ not implemented",
                ))
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("fill_ not implemented for sparse"))
        }
    }

    #[pyo3(name = "add_")]
    pub fn add_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::add_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::add_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::add_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("add_ not implemented for sparse"))
        }
    }

    #[pyo3(name = "mul_")]
    pub fn mul_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::mul_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::mul_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::mul_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("mul_ not implemented for sparse"))
        }
    }

    #[pyo3(name = "sub_")]
    pub fn sub_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::sub_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::sub_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::sub_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("sub_ not implemented for sparse"))
        }
    }

    #[pyo3(name = "div_")]
    pub fn div_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => tensor::ops::div_(inner, Float32(value as f32)).map_err(to_py_err),
            TensorWrapper::CpuDenseF64(inner) => tensor::ops::div_(inner, Float64(value)).map_err(to_py_err),
            TensorWrapper::CpuDenseI64(inner) => tensor::ops::div_(inner, dtype::int::Int64(value as i64)).map_err(to_py_err),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("div_ not implemented for sparse"))
        }
    }

    pub fn abs_(&mut self) -> PyResult<()> { dispatch_tensor_mut!(self, inner => tensor::ops::abs_(inner).map_err(to_py_err)); Ok(()) }
    pub fn neg_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::neg_(inner).map_err(to_py_err)); Ok(()) }
    pub fn sin_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::sin_(inner).map_err(to_py_err)); Ok(()) }
    pub fn cos_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::cos_(inner).map_err(to_py_err)); Ok(()) }
    pub fn tan_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::tan_(inner).map_err(to_py_err)); Ok(()) }
    pub fn asin_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::asin_(inner).map_err(to_py_err)); Ok(()) }
    pub fn acos_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::acos_(inner).map_err(to_py_err)); Ok(()) }
    pub fn atan_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::atan_(inner).map_err(to_py_err)); Ok(()) }
    pub fn sinh_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::sinh_(inner).map_err(to_py_err)); Ok(()) }
    pub fn cosh_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::cosh_(inner).map_err(to_py_err)); Ok(()) }
    pub fn tanh_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::tanh_(inner).map_err(to_py_err)); Ok(()) }
    pub fn exp_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::exp_(inner).map_err(to_py_err)); Ok(()) }
    pub fn log_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::log_(inner).map_err(to_py_err)); Ok(()) }
    pub fn sqrt_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::sqrt_(inner).map_err(to_py_err)); Ok(()) }
    pub fn ceil_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::ceil_(inner).map_err(to_py_err)); Ok(()) }
    pub fn floor_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::floor_(inner).map_err(to_py_err)); Ok(()) }
    pub fn round_(&mut self) -> PyResult<()> { crate::dispatch_float_tensor_mut!(self, inner => tensor::ops::inplace::round_(inner).map_err(to_py_err)); Ok(()) }
}
