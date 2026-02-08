use crate::{dispatch_tensor_mut, dispatch_float_tensor, dispatch_ord_tensor, dispatch_ord_tensor_mut, dispatch_unary, dispatch_float_unary, dispatch_unary_inplace, dispatch_float_unary_inplace, dispatch_float_binary};
use crate::tensor::class::{PyTensor, to_py_err};
use crate::tensor::wrapper::{TensorWrapper, WrapTensor};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
use dtype::int::Int64;

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

    pub fn atan2(&self, other: &PyTensor) -> PyResult<PyTensor> {
        dispatch_float_binary!(self, other, a, b => {
            let a_strided = a.to_strided().map_err(to_py_err)?;
            let b_strided = b.to_strided().map_err(to_py_err)?;
            let res = tensor::ops::atan2(&a_strided, &b_strided).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn clamp(&self, min: f64, max: f64) -> PyResult<PyTensor> {
        use dtype::float::{Float32, Float64};
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.clamp(Float32::new(min as f32), Float32::new(max as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.clamp(Float64::new(min), Float64::new(max)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let res = t.clamp(Float32::new(min as f32), Float32::new(max as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("clamp not implemented for this tensor type")),
        }
    }

    pub fn nan_to_num(&self, nan: Option<f64>, posinf: Option<f64>, neginf: Option<f64>) -> PyResult<PyTensor> {
        use dtype::float::{Float32, Float64};
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
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("nan_to_num not implemented for this tensor type")),
        }
    }

    pub fn renorm(&self, p: f64, dim: i64, maxnorm: f64) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
            let rank = a.shape().ndim();
            let dim_idx = if dim < 0 { (rank as i64 + dim) as usize } else { dim as usize };

            use num_traits::NumCast;
            let p_casted = NumCast::from(p).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("renorm: failed to cast p"))?;
            let maxnorm_casted = NumCast::from(maxnorm).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("renorm: failed to cast maxnorm"))?;

            let res = tensor::ops::renorm(a, p_casted, dim_idx, maxnorm_casted).map_err(to_py_err)?;
            Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn clamp_(&mut self, min: f64, max: f64) -> PyResult<PyTensor> {
        dispatch_ord_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let min_casted = NumCast::from(min).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_: failed to cast min"))?;
            let max_casted = NumCast::from(max).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_: failed to cast max"))?;
            tensor::ops::clamp_(a, min_casted, max_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn clamp_min_(&mut self, min: f64) -> PyResult<PyTensor> {
        dispatch_ord_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let min_casted = NumCast::from(min).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_min_: failed to cast min"))?;
            tensor::ops::clamp_min_(a, min_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn clamp_max_(&mut self, max: f64) -> PyResult<PyTensor> {
        dispatch_ord_tensor_mut!(self, a => {
            use num_traits::NumCast;
            let max_casted = NumCast::from(max).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("clamp_max_: failed to cast max"))?;
            tensor::ops::clamp_max_(a, max_casted).map_err(to_py_err)?;
            Ok(self.clone())
        })
    }

    pub fn sort(&self, dim: usize, descending: bool) -> PyResult<(PyTensor, PyTensor)> {
        dispatch_ord_tensor!(self, a => {
            let (values, indices) = tensor::ops::sort(a, dim, descending).map_err(to_py_err)?;
            let indices_i64: Vec<Int64> = indices.into_iter().map(|i| Int64::new(i as i64)).collect();
            let indices_tensor = tensor::Tensor::<backend::CpuBackend<Int64>, storage::DenseStorage<Int64>, Int64>::from_vec(indices_i64, values.shape().dims()).map_err(to_py_err)?;
            Ok((PyTensor { inner: values.wrap() }, PyTensor { inner: indices_tensor.wrap() }))
        })
    }

    pub fn topk(&self, k: usize, dim: usize, largest: bool) -> PyResult<(PyTensor, PyTensor)> {
        dispatch_ord_tensor!(self, a => {
            let (values, indices) = tensor::ops::topk(a, k, dim, largest).map_err(to_py_err)?;
            let indices_i64: Vec<Int64> = indices.into_iter().map(|i| Int64::new(i as i64)).collect();
            let indices_tensor = tensor::Tensor::<backend::CpuBackend<Int64>, storage::DenseStorage<Int64>, Int64>::from_vec(indices_i64, values.shape().dims()).map_err(to_py_err)?;
            Ok((PyTensor { inner: values.wrap() }, PyTensor { inner: indices_tensor.wrap() }))
        })
    }

    pub fn unique(&self, _sorted: bool, _return_inverse: bool, _return_counts: bool, _dim: Option<usize>) -> PyResult<PyTensor> {
        dispatch_ord_tensor!(self, a => {
             let res = tensor::ops::unique(a).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }

    pub fn square(&self) -> PyResult<PyTensor> {
        dispatch_float_tensor!(self, a => {
             let res = tensor::ops::square(a).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    }

    // Aliases for PyTorch compatibility
    pub fn absolute(&self) -> PyResult<PyTensor> {
        self.abs()
    }
    
    pub fn arccos(&self) -> PyResult<PyTensor> {
        self.acos()
    }

    pub fn arcsin(&self) -> PyResult<PyTensor> {
        self.asin()
    }
    
    pub fn arctan(&self) -> PyResult<PyTensor> {
        self.atan()
    }

    pub fn clamp_min(&self, min: f64) -> PyResult<PyTensor> {
        use dtype::float::{Float32, Float64};
         match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.clamp_min(Float32::new(min as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.clamp_min(Float64::new(min)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                 let res = t.clamp_min(Float32::new(min as f32)).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("clamp_min not implemented for this tensor type")),
        }
    }
    
    pub fn clamp_max(&self, max: f64) -> PyResult<PyTensor> {
        use dtype::float::{Float32, Float64};
         match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = t.clamp_max(Float32::new(max as f32)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = t.clamp_max(Float64::new(max)).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                 let res = t.clamp_max(Float32::new(max as f32)).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("clamp_max not implemented for this tensor type")),
        }
    }

    // In-place operations
    pub fn abs_(&mut self) -> PyResult<PyTensor> {
        dispatch_ord_tensor_mut!(self, inner => tensor::ops::abs_(inner).map_err(to_py_err))?;
        Ok(self.clone())
    }

    pub fn acos_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::acos_(inner))?;
        Ok(self.clone())
    }

    pub fn asin_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::asin_(inner))?;
        Ok(self.clone())
    }

    pub fn atan_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::atan_(inner))?;
        Ok(self.clone())
    }

    pub fn ceil_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::ceil_(inner))?;
        Ok(self.clone())
    }

    pub fn cos_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::cos_(inner))?;
        Ok(self.clone())
    }

    pub fn cosh_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::cosh_(inner))?;
        Ok(self.clone())
    }

    pub fn exp_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::exp_(inner))?;
        Ok(self.clone())
    }

    pub fn floor_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::floor_(inner))?;
        Ok(self.clone())
    }

    pub fn log_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::log_(inner))?;
        Ok(self.clone())
    }

    pub fn log10_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::log10_(inner))?;
        Ok(self.clone())
    }

    pub fn log2_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::log2_(inner))?;
        Ok(self.clone())
    }

    pub fn neg_(&mut self) -> PyResult<PyTensor> {
         dispatch_tensor_mut!(self, inner => tensor::ops::neg_(inner).map_err(to_py_err))?;
         Ok(self.clone())
    }

    pub fn round_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::round_(inner))?;
        Ok(self.clone())
    }

    pub fn rsqrt_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::rsqrt_(inner))?;
        Ok(self.clone())
    }

    pub fn sigmoid_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::sigmoid_(inner))?;
        Ok(self.clone())
    }

    pub fn sin_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::sin_(inner))?;
        Ok(self.clone())
    }

    pub fn sinh_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::sinh_(inner))?;
        Ok(self.clone())
    }

    pub fn sqrt_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::sqrt_(inner))?;
        Ok(self.clone())
    }

    pub fn tan_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::tan_(inner))?;
        Ok(self.clone())
    }

    pub fn tanh_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::tanh_(inner))?;
        Ok(self.clone())
    }

    pub fn trunc_(&mut self) -> PyResult<PyTensor> {
        dispatch_float_unary_inplace!(self, inner => tensor::ops::trunc_(inner))?;
        Ok(self.clone())
    }}
