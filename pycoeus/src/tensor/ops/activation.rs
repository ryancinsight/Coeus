use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::{dispatch_unary, dispatch_float_unary};
use pyo3::prelude::*;
use dtype::float::{Float32, Float64};

#[pymethods]
impl PyTensor {
    pub fn neg(&self) -> PyResult<PyTensor> {
        dispatch_unary!(self, inner => tensor::ops::neg(inner))
    }

    pub fn abs(&self) -> PyResult<PyTensor> {
        dispatch_unary!(self, inner => tensor::ops::abs(inner))
    }

    pub fn exp(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::exp(inner))
    }

    pub fn log(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::log(inner))
    }

    pub fn sqrt(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::sqrt(inner))
    }

    pub fn sin(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::sin(inner))
    }

    pub fn cos(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::cos(inner))
    }

    pub fn tan(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::tan(inner))
    }

    pub fn asin(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::asin(inner))
    }

    pub fn acos(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::acos(inner))
    }

    pub fn atan(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::atan(inner))
    }

    pub fn sinh(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::sinh(inner))
    }

    pub fn cosh(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::cosh(inner))
    }

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

    pub fn asinh(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::asinh(inner))
    }

    pub fn acosh(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::acosh(inner))
    }

    pub fn atanh(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::atanh(inner))
    }

    pub fn sign(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::sign(inner))
    }

    pub fn signbit(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::signbit(inner))
    }

    pub fn trunc(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::trunc(inner))
    }

    pub fn frac(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::frac(inner))
    }

    pub fn erfc(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::erfc(inner))
    }

    pub fn erfinv(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::erfinv(inner))
    }

    pub fn reciprocal(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::reciprocal(inner))
    }

    pub fn expm1(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::expm1(inner))
    }

    pub fn log1p(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::log1p(inner))
    }

    pub fn deg2rad(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::deg2rad(inner))
    }

    pub fn rad2deg(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::rad2deg(inner))
    }

    pub fn ceil(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::ceil(inner))
    }

    pub fn floor(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::floor(inner))
    }

    pub fn round(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::round(inner))
    }

    pub fn rsqrt(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::rsqrt(inner))
    }

    pub fn log2(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::log2(inner))
    }

    pub fn log10(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::log10(inner))
    }

    pub fn exp2(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::exp2(inner))
    }

    pub fn erf(&self) -> PyResult<PyTensor> {
        dispatch_float_unary!(self, inner => tensor::ops::erf(inner))
    }

    pub fn clamp(&self, min: f64, max: f64) -> PyResult<PyTensor> {


        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t
                    .clamp(Float32::new(min as f32), Float32::new(max as f32))
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t
                    .clamp(Float64::new(min), Float64::new(max))
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t
                    .clamp(Float32::new(min as f32), Float32::new(max as f32))
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "clamp not implemented for this tensor type",
            )),
        }
    }

    /// Replace NaN, positive infinity, and negative infinity values.

    pub fn nan_to_num(&self, nan: Option<f64>, posinf: Option<f64>, neginf: Option<f64>) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::nan_to_num(
                    t,
                    nan.map(|v| Float32::new(v as f32)),
                    posinf.map(|v| Float32::new(v as f32)),
                    neginf.map(|v| Float32::new(v as f32)),
                ).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::nan_to_num(
                    t,
                    nan.map(Float64::new),
                    posinf.map(Float64::new),
                    neginf.map(Float64::new),
                ).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "nan_to_num not implemented for this tensor type",
            )),
        }
    }
}
