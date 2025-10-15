//! Error types for distributed training operations

use thiserror::Error;

/// Result type alias for distributed operations
pub type Result<T> = std::result::Result<T, DistributedError>;

/// Errors that can occur during distributed training operations
#[derive(Error, Debug)]
pub enum DistributedError {
    #[error("Communication error: {message}")]
    Communication { message: String },

    #[error("Gradient synchronization failed: {message}")]
    GradientSync { message: String },

    #[error("Invalid process group configuration: {message}")]
    ProcessGroupConfig { message: String },

    #[error("Device mismatch: expected {expected}, got {actual}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("Tensor shape mismatch in reduction: {message}")]
    ShapeMismatch { message: String },

    #[error("Operation timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Backend not available: {backend}")]
    BackendUnavailable { backend: String },

    #[error("Gradient buffer overflow: required {required}, available {available}")]
    BufferOverflow { required: usize, available: usize },
}
