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

    #[error("Tensor operation failed: {source}")]
    TensorError { source: Box<tensor::TensorError> },

    #[error("Optimizer operation failed: {source}")]
    OptimizerError {
        source: Box<optim::error::OptimError>,
    },

    #[error("Gradient buffer overflow: required {required}, available {available}")]
    BufferOverflow { required: usize, available: usize },
}

impl From<tensor::TensorError> for DistributedError {
    fn from(error: tensor::TensorError) -> Self {
        DistributedError::TensorError {
            source: Box::new(error),
        }
    }
}

impl From<optim::error::OptimError> for DistributedError {
    fn from(error: optim::error::OptimError) -> Self {
        DistributedError::OptimizerError {
            source: Box::new(error),
        }
    }
}

impl From<std::io::Error> for DistributedError {
    fn from(error: std::io::Error) -> Self {
        DistributedError::Communication {
            message: format!("I/O error: {}", error),
        }
    }
}
