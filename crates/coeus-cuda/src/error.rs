use coeus_core::BackendError;
use thiserror::Error;

/// Failure returned by CUDA elementwise dispatch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CudaBackendError {
    /// The CUDA provider only has a native scan kernel for rank-two layouts.
    #[error("CUDA {operation} does not support layout rank {rank}; maximum rank is {max_rank}")]
    UnsupportedRank {
        /// Operation family that rejected the layout.
        operation: &'static str,
        /// Requested layout rank.
        rank: usize,
        /// Largest rank implemented by the native provider path.
        max_rank: usize,
    },
    /// The scan layouts do not satisfy the same-shape backend contract.
    #[error("CUDA {operation} rejected its layouts: {reason}")]
    InvalidLayout {
        /// Operation family that rejected the layout.
        operation: &'static str,
        /// Backend-independent layout invariant that failed.
        reason: &'static str,
    },
    /// The requested operation requires a live CUDA provider context.
    #[error("CUDA {operation} provider is unavailable: {reason}")]
    ProviderUnavailable {
        /// Operation that requires the CUDA provider.
        operation: &'static str,
        /// Provider capability that is unavailable.
        reason: &'static str,
    },
    /// The CUDA provider rejected a submitted kernel.
    #[error("CUDA {operation} dispatch failed: {source}")]
    Dispatch {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Provider failure.
        #[source]
        source: hephaestus_cuda::HephaestusError,
    },
    /// The explicit CPU capability boundary failed.
    #[error("CUDA {operation} CPU capability path failed: {source}")]
    CpuCapability {
        /// Operation family being executed by the capability path.
        operation: &'static str,
        /// CPU operation failure.
        #[source]
        source: BackendError,
    },
}

impl From<BackendError> for CudaBackendError {
    fn from(source: BackendError) -> Self {
        Self::cpu_capability("elementwise", source)
    }
}

impl CudaBackendError {
    pub(crate) fn cpu_capability(operation: &'static str, source: BackendError) -> Self {
        Self::CpuCapability { operation, source }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dispatch(
        operation: &'static str,
        source: hephaestus_cuda::HephaestusError,
    ) -> Self {
        Self::Dispatch { operation, source }
    }
}
