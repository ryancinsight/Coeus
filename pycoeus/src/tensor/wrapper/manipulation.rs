use crate::tensor::class::TensorWrapper;
use crate::tensor::wrapper::WrapTensor;
use tensor::TensorError;

impl TensorWrapper {
    pub fn permute(&self, dims: &[usize]) -> Result<Self, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let res = ::tensor::ops::permute(t, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            Self::CpuDenseF64(t) => {
                let res = ::tensor::ops::permute(t, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            Self::CpuDenseI64(t) => {
                let res = ::tensor::ops::permute(t, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            Self::CpuStridedF32(t) => {
                let dense = t.to_cpu_dense().map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                let res = ::tensor::ops::permute(&dense, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            Self::CpuStridedF64(t) => {
                let dense = t.to_cpu_dense().map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                let res = ::tensor::ops::permute(&dense, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            Self::CpuStridedI64(t) => {
                let dense = t.to_cpu_dense().map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                let res = ::tensor::ops::permute(&dense, dims).map_err(|e| TensorError::BackendError(format!(" {}", e)))?;
                Ok(res.wrap())
            }
            _ => Err(TensorError::BackendError("Permute not implemented for this storage".to_string())),
        }
    }
}
