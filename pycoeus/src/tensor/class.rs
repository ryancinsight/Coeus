use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use coeus_linalg::qr::QRResult;
use coeus_linalg::svd::SVDResult;
use coeus_linalg::{Cholesky, Det, Inverse, Norm, Solve, QR, SVD};
use dtype::float::{Float32, Float64};
use dtype::int::Int64;
use pyo3::prelude::*;
use storage::{CsrStorage, DenseStorage};
use tensor::tensor_core::Tensor;
use tensor::TensorError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(name = "Device")]
pub enum Device {
    CPU,
    CUDA,
}

#[derive(Clone)]
pub enum TensorWrapper {
    CpuDenseF32(Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuDenseF64(Tensor<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuDenseF32(Tensor<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuSparseF32(Tensor<CpuBackend<Float32>, CsrStorage<Float32>, Float32>),
    CpuSparseF64(Tensor<CpuBackend<Float64>, CsrStorage<Float64>, Float64>),
    CpuDenseI64(Tensor<CpuBackend<Int64>, DenseStorage<Int64>, Int64>),
}

impl TensorWrapper {
    pub fn shape(&self) -> &::tensor::Shape {
        match self {
            Self::CpuDenseF32(t) => t.shape(),
            Self::CpuDenseF64(t) => t.shape(),
            Self::CpuSparseF32(t) => t.shape(),
            Self::CpuSparseF64(t) => t.shape(),
            Self::CpuDenseI64(t) => t.shape(),
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => t.shape(),
        }
    }

    pub fn requires_grad_(self, requires_grad: bool) -> Self {
        match self {
            Self::CpuDenseF32(t) => Self::CpuDenseF32(t.requires_grad_(requires_grad)),
            Self::CpuDenseF64(t) => Self::CpuDenseF64(t.requires_grad_(requires_grad)),
            Self::CpuSparseF32(t) => Self::CpuSparseF32(t.requires_grad_(requires_grad)),
            Self::CpuSparseF64(t) => Self::CpuSparseF64(t.requires_grad_(requires_grad)),
            Self::CpuDenseI64(t) => Self::CpuDenseI64(t.requires_grad_(requires_grad)),
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => Self::GpuDenseF32(t.requires_grad_(requires_grad)),
        }
    }

    pub fn len(&self) -> usize {
        self.shape().size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[Float32] {
        match self {
            Self::CpuDenseF32(t) => t.as_slice(),
            _ => panic!("as_slice only supported for CpuDenseF32 in current FFT impl"),
        }
    }

    pub fn storage_ref(&self) -> &DenseStorage<Float32> {
        match self {
            Self::CpuDenseF32(t) => t.storage(),
            _ => panic!("storage_ref only supported for CpuDenseF32 in current FFT impl"),
        }
    }

    pub fn inv(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .inv()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = t
                    .inv()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            _ => panic!("inv not implemented for this tensor type"),
        }
    }

    pub fn norm(&self) -> Result<Float32, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .norm()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(res)
            }
            _ => panic!("norm only implemented for F32"),
        }
    }

    pub fn norm_p(&self, p: Float32) -> Result<Float32, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .norm_p(p)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(res)
            }
            _ => panic!("norm_p only implemented for F32"),
        }
    }

    pub fn det(&self) -> Result<Float32, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .det()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(res)
            }
            _ => panic!("det only implemented for F32"),
        }
    }

    pub fn solve(&self, other: &Self) -> Result<Self, TensorError> {
        match (self, other) {
            (Self::CpuDenseF32(a), Self::CpuDenseF32(b)) => {
                let res = a
                    .solve(b)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            _ => panic!("solve only implemented for F32"),
        }
    }

    pub fn cholesky(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .cholesky()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            _ => panic!("cholesky only implemented for F32"),
        }
    }

    pub fn qr(&self) -> Result<QRResult<CpuBackend<Float32>, Float32>, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .qr()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(res)
            }
            _ => panic!("qr only implemented for F32"),
        }
    }

    pub fn svd(
        &self,
        full_matrices: bool,
    ) -> Result<SVDResult<CpuBackend<Float32>, Float32>, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .svd(full_matrices)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(res)
            }
            _ => panic!("svd only implemented for F32"),
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = t
                    .permute(dims)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = t
                    .permute(dims)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = t
                    .permute(dims)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "permute not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Element-wise reciprocal (1/x)
    pub fn reciprocal(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::reciprocal(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::reciprocal(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::reciprocal(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "reciprocal not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Element-wise exp(x) - 1
    pub fn expm1(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::expm1(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::expm1(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::expm1(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "expm1 not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Element-wise log(1 + x)
    pub fn log1p(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::log1p(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::log1p(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::log1p(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "log1p not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Cumulative sum along a dimension
    pub fn cumsum(&self, dim: usize) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::cumsum(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::cumsum(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::cumsum(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "cumsum not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Cumulative product along a dimension
    pub fn cumprod(&self, dim: usize) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::cumprod(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::cumprod(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::cumprod(t, dim)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "cumprod not implemented for sparse tensors".to_string(),
            )),
        }
    }

    /// Degrees to radians
    pub fn deg2rad(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::deg2rad(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::deg2rad(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::deg2rad(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "deg2rad not implemented for sparse or integer tensors".to_string(),
            )),
        }

    }

    /// Radians to degrees
    pub fn rad2deg(&self) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = tensor::ops::rad2deg(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF32(res))
            }
            Self::CpuDenseF64(t) => {
                let res = tensor::ops::rad2deg(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::CpuDenseF64(res))
            }
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => {
                let res = tensor::ops::rad2deg(t)
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(Self::GpuDenseF32(res))
            }
            Self::CpuSparseF32(_) | Self::CpuSparseF64(_) | Self::CpuDenseI64(_) => Err(TensorError::BackendError(
                "rad2deg not implemented for sparse or integer tensors".to_string(),
            )),
        }

    }
}

pub fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    crate::error::convert_error(format!("Tensor error: {}", e))
}

#[macro_export]
macro_rules! dispatch_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuDenseI64($inner) => $expr,
        }
    };
}


#[macro_export]
macro_rules! dispatch_tensor_mut {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &mut $tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuDenseI64($inner) => $expr,
        }
    };
}

#[macro_export]
macro_rules! dispatch_float_tensor_mut {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &mut $tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuDenseI64(_) => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Operation not implemented for integer tensors"
                ))
            }
        }
    };
}


#[macro_export]
macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, $a:ident, $b:ident => $expr:expr) => {
        match (&$lhs.inner, &$rhs.inner) {
            (TensorWrapper::CpuDenseF32($a), TensorWrapper::CpuDenseF32($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (TensorWrapper::CpuDenseF64($a), TensorWrapper::CpuDenseF64($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (TensorWrapper::GpuDenseF32($a), TensorWrapper::GpuDenseF32($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            (TensorWrapper::CpuSparseF32($a), TensorWrapper::CpuSparseF32($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF32(res),
                })
            }
            (TensorWrapper::CpuSparseF64($a), TensorWrapper::CpuSparseF64($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF64(res),
                })
            }
            (TensorWrapper::CpuDenseI64($a), TensorWrapper::CpuDenseI64($b)) => {
                let res = $expr;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseI64(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Tensor backend/dtype/storage mismatch",
            )),
        }
    };
}

#[macro_export]
macro_rules! dispatch_unary {
    ($self:expr, $inner:ident => $expr:expr) => {
        match &$self.inner {
            TensorWrapper::CpuDenseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            TensorWrapper::CpuSparseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF32(res),
                })
            }
            TensorWrapper::CpuSparseF64($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF64(res),
                })
            }
            TensorWrapper::CpuDenseI64($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseI64(res),
                })
            }
        }
    };
}

#[macro_export]
macro_rules! dispatch_float_unary {
    ($self:expr, $inner:ident => $expr:expr) => {
        match &$self.inner {
            TensorWrapper::CpuDenseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            TensorWrapper::CpuSparseF32($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF32(res),
                })
            }
            TensorWrapper::CpuSparseF64($inner) => {
                let res = $expr.map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuSparseF64(res),
                })
            }
            TensorWrapper::CpuDenseI64(_) => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Operation not implemented for integer tensors"
                ))
            }
        }
    };
}

#[pyclass(name = "Tensor", module = "coeus", subclass)]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: TensorWrapper,
}
