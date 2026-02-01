use super::TensorWrapper;
use tensor::TensorError;

impl TensorWrapper {
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
            _ => Err(TensorError::BackendError(
                "reciprocal not implemented for sparse or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "expm1 not implemented for sparse or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "log1p not implemented for sparse or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "cumsum not implemented for sparse or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "cumprod not implemented for sparse or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "deg2rad not implemented for sparse, integer or complex tensors".to_string(),
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
            _ => Err(TensorError::BackendError(
                "rad2deg not implemented for sparse, integer or complex tensors".to_string(),
            )),
        }
    }
}
