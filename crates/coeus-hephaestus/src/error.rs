use coeus_core::BackendError;
use hephaestus_core::HephaestusError;
use std::{fmt, sync::Arc};
use thiserror::Error;

/// Shared provider error retained by a cached device-initialization result.
///
/// The cache stores one error instance so failed initialization is stable
/// across callers without requiring the provider error to implement `Clone`.
#[derive(Debug, Clone)]
pub struct SharedHephaestusError(Arc<HephaestusError>);

impl SharedHephaestusError {
    /// Wrap a provider error for storage in a cached initialization result.
    #[must_use]
    pub fn new(source: HephaestusError) -> Self {
        Self(Arc::new(source))
    }
}

impl fmt::Display for SharedHephaestusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for SharedHephaestusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

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
    /// The selected provider could not initialize its device.
    #[error("{operation} Hephaestus provider initialization failed: {source}")]
    Initialization {
        /// Operation that required provider initialization.
        operation: &'static str,
        /// Cached provider failure with its source chain preserved.
        #[source]
        source: SharedHephaestusError,
    },
}

impl HephaestusBackendError {
    pub(crate) fn device(operation: &'static str, source: HephaestusError) -> Self {
        Self::Device { operation, source }
    }

    /// Build an error for a cached provider initialization failure.
    #[must_use]
    pub fn initialization(operation: &'static str, source: SharedHephaestusError) -> Self {
        Self::Initialization { operation, source }
    }
}
