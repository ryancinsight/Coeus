use coeus_core::BackendError;
use thiserror::Error;

/// Failure returned by CUDA backend operation and provider dispatch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CudaBackendError {
    /// The operation rejected a backend-independent contract before dispatch.
    #[error("CUDA operation validation failed: {source}")]
    Validation {
        /// Backend-independent validation failure.
        #[source]
        source: BackendError,
    },
    /// The CUDA provider supports reduction and scan layouts up to rank two.
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
    /// A native CUDA kernel could not be validated or launched.
    #[error("CUDA {operation} kernel failed: {reason}")]
    Kernel {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Stable reason for the rejected launch.
        reason: &'static str,
    },
}

impl From<BackendError> for CudaBackendError {
    fn from(source: BackendError) -> Self {
        Self::validation(source)
    }
}

impl From<coeus_hephaestus::HephaestusBackendError> for CudaBackendError {
    fn from(source: coeus_hephaestus::HephaestusBackendError) -> Self {
        match source {
            // The historical CUDA backend reported rank rejection for axis
            // reductions under the operation label "reduction"; the bridge
            // labels the same dispatch "reduce". Normalize so the public
            // `UnsupportedRank` contract is unchanged for consumers.
            coeus_hephaestus::HephaestusBackendError::Backend(BackendError::UnsupportedRank {
                operation: "reduce",
                rank,
                max_rank,
            }) => Self::UnsupportedRank {
                operation: "reduction",
                rank,
                max_rank,
            },
            coeus_hephaestus::HephaestusBackendError::Backend(BackendError::UnsupportedRank {
                operation,
                rank,
                max_rank,
            }) => Self::UnsupportedRank {
                operation,
                rank,
                max_rank,
            },
            coeus_hephaestus::HephaestusBackendError::Backend(error) => Self::validation(error),
            coeus_hephaestus::HephaestusBackendError::Device { operation, source } => {
                Self::Dispatch { operation, source }
            }
            // The bridge error is non-exhaustive so future provider categories
            // degrade to a typed backend validation failure rather than panic.
            _ => Self::validation(BackendError::Storage {
                operation: "hephaestus dispatch",
                reason: "unclassified Hephaestus backend failure".to_string(),
            }),
        }
    }
}

impl CudaBackendError {
    pub(crate) fn validation(source: BackendError) -> Self {
        Self::Validation { source }
    }

    // Both callers sit under `#[cfg(not(feature = "cuda"))]`: with the
    // provider built there is a real dispatch path and nothing constructs
    // this. Ungated it was dead code in the configuration CI builds.
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn kernel(operation: &'static str, reason: &'static str) -> Self {
        Self::Kernel { operation, reason }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dispatch(
        operation: &'static str,
        source: hephaestus_cuda::HephaestusError,
    ) -> Self {
        Self::Dispatch { operation, source }
    }
}
