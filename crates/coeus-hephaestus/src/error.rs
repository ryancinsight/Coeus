use coeus_core::BackendError;
use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Error returned by the generic Coeus-Hephaestus integration boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HephaestusBackendError {
    /// A Coeus layout or storage contract rejected the operation.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// The selected Hephaestus provider rejected device execution or transfer.
    #[error("{operation} Hephaestus dispatch failed: {source}")]
    Device {
        /// Operation family that reached the provider.
        operation: &'static str,
        /// Provider error with its original failure category preserved.
        #[source]
        source: HephaestusError,
    },
}

impl HephaestusBackendError {
    /// Preserve a provider failure with its Coeus operation context.
    #[must_use]
    pub fn device(operation: &'static str, source: HephaestusError) -> Self {
        Self::Device { operation, source }
    }
}
