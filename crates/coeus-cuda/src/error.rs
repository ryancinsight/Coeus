use coeus_core::BackendError;
use thiserror::Error;

/// Failure returned by CUDA elementwise dispatch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CudaBackendError {
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
