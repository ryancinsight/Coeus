use super::TensorWrapper;
use dtype::float::Float32;
use storage::DenseStorage;

impl TensorWrapper {
    pub fn shape(&self) -> &::tensor::Shape {
        match self {
            Self::CpuDenseF32(t) => t.shape(),
            Self::CpuDenseF64(t) => t.shape(),
            Self::CpuSparseF32(t) => t.shape(),
            Self::CpuSparseF64(t) => t.shape(),
            Self::CpuDenseI64(t) => t.shape(),
            Self::CpuDenseC32(t) => t.shape(),
            Self::CpuStridedF32(t) => t.shape(),
            Self::CpuStridedF64(t) => t.shape(),
            Self::CpuStridedI64(t) => t.shape(),
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => t.shape(),
            #[cfg(feature = "gpu")]
            Self::GpuStridedF32(t) => t.shape(),
            Self::CpuStridedC32(t) => t.shape(),
        }
    }

    pub fn requires_grad_(self, requires_grad: bool) -> Self {
        match self {
            Self::CpuDenseF32(t) => Self::CpuDenseF32(t.requires_grad_(requires_grad)),
            Self::CpuDenseF64(t) => Self::CpuDenseF64(t.requires_grad_(requires_grad)),
            Self::CpuSparseF32(t) => Self::CpuSparseF32(t.requires_grad_(requires_grad)),
            Self::CpuSparseF64(t) => Self::CpuSparseF64(t.requires_grad_(requires_grad)),
            Self::CpuDenseI64(t) => Self::CpuDenseI64(t.requires_grad_(requires_grad)),
            Self::CpuDenseC32(t) => Self::CpuDenseC32(t.requires_grad_(requires_grad)),
            Self::CpuStridedF32(t) => Self::CpuStridedF32(t.requires_grad_(requires_grad)),
            Self::CpuStridedF64(t) => Self::CpuStridedF64(t.requires_grad_(requires_grad)),
            Self::CpuStridedI64(t) => Self::CpuStridedI64(t.requires_grad_(requires_grad)),
            #[cfg(feature = "gpu")]
            Self::GpuDenseF32(t) => Self::GpuDenseF32(t.requires_grad_(requires_grad)),
            #[cfg(feature = "gpu")]
            Self::GpuStridedF32(t) => Self::GpuStridedF32(t.requires_grad_(requires_grad)),
            Self::CpuStridedC32(t) => Self::CpuStridedC32(t.requires_grad_(requires_grad)),
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
}
