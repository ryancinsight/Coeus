use coeus_core::BackendError;
use coeus_hephaestus::SharedHephaestusError;
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
    /// The CUDA provider rejected allocation, transfer, or device binding.
    #[error("CUDA {operation} provider failure: {source}")]
    Provider {
        /// Backend operation that reached the provider.
        operation: &'static str,
        /// Provider failure with its original category preserved.
        #[source]
        source: hephaestus_cuda::HephaestusError,
    },
    /// The CUDA provider could not initialize a device.
    #[error("CUDA provider initialization failed: {source}")]
    Initialization {
        /// Cached provider failure with its source chain preserved.
        #[source]
        source: SharedHephaestusError,
    },
    /// A CUDA-only operation could not be dispatched by the selected device
    /// path. The caller must not replace this failure with a host execution
    /// path because that changes the backend contract.
    #[error("CUDA {operation} dispatch unavailable: {reason}")]
    DispatchUnavailable {
        /// Operation family that could not be dispatched.
        operation: &'static str,
        /// Stable reason identifying the rejected capability boundary.
        reason: &'static str,
    },
}

impl From<BackendError> for CudaBackendError {
    fn from(source: BackendError) -> Self {
        Self::cpu_capability("elementwise", source)
    }
}

impl CudaBackendError {
    pub(crate) fn initialization(source: SharedHephaestusError) -> Self {
        Self::Initialization { source }
    }

    pub(crate) fn cpu_capability(operation: &'static str, source: BackendError) -> Self {
        Self::CpuCapability { operation, source }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dispatch_unavailable(operation: &'static str, reason: &'static str) -> Self {
        Self::DispatchUnavailable { operation, reason }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn provider(
        operation: &'static str,
        source: hephaestus_cuda::HephaestusError,
    ) -> Self {
        Self::Provider { operation, source }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dispatch(
        operation: &'static str,
        source: hephaestus_cuda::HephaestusError,
    ) -> Self {
        Self::Dispatch { operation, source }
    }
}
