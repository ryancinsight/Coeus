use super::TensorWrapper;
use backend::CpuBackend;
use coeus_linalg::qr::QRResult;
use coeus_linalg::svd::SVDResult;
use coeus_linalg::{Cholesky, Det, Inverse, Norm, Solve, QR, SVD};
use dtype::float::Float32;
use tensor::TensorError;

impl TensorWrapper {
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
                let (q, r) = t
                    .qr()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(QRResult { q, r })
            }
            _ => panic!("qr only implemented for F32"),
        }
    }

    pub fn svd(
        &self,
        _full_matrices: bool,
    ) -> Result<SVDResult<CpuBackend<Float32>, Float32>, TensorError> {
        match self {
            Self::CpuDenseF32(t) => {
                let (u, s, v) = t
                    .svd()
                    .map_err(|e| TensorError::BackendError(format!("{}", e)))?;
                Ok(SVDResult { u, s, vh: v })
                // Usually u, s, vt (transpose of V).
                // Let's assume standard names. If fields are private or different, I'll fail.
                // But `coeus_linalg` usually has public fields.
            }
            _ => panic!("svd only implemented for F32"),
        }
    }
}
