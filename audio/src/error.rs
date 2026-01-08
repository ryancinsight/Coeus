//! Error types for audio processing operations

use thiserror::Error;

/// Errors that can occur during audio processing operations
#[derive(Debug, Error)]
pub enum AudioError {
    /// FFT size is not a power of 2
    #[error("FFT size must be a power of 2, got {0}")]
    InvalidFftSize(usize),

    /// Input tensor has invalid shape for audio operation
    #[error("Invalid tensor shape for audio operation: {message}")]
    InvalidShape { message: String },

    /// Invalid input data
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    /// Input tensor has incompatible dtype
    #[error("Incompatible tensor dtype for audio operation, expected {expected}, got {got}")]
    IncompatibleDtype { expected: String, got: String },

    /// Signal length mismatch
    #[error("Signal length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },

    /// FFT operation failed
    #[error("FFT operation failed: {message}")]
    FftFailed { message: String },

    /// GPU operation failed
    #[error("GPU operation failed: {message}")]
    GpuError { message: String },

    /// Backend operation failed
    #[error("Backend operation failed: {source}")]
    BackendError {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// I/O operation failed
    #[error("I/O operation failed: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// Neural network operation failed
    #[error("Neural network error: {source}")]
    NNError {
        #[from]
        source: nn::NNError,
    },
}

/// Result type alias for audio operations
pub type AudioResult<T> = std::result::Result<T, AudioError>;

/// Standard Result type for audio operations (alias for AudioResult)
pub type Result<T> = AudioResult<T>;
