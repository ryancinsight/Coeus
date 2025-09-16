//! Backend-specific error types and results

/// Result type for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

/// Errors that can occur during backend operations
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Device mismatch: operation requires {required:?}, got {actual:?}")]
    DeviceMismatch {
        required: crate::Device,
        actual: crate::Device,
    },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DtypeMismatch { expected: String, actual: String },

    #[error("GPU operation failed: {message}")]
    GpuError { message: String },

    #[error("Memory allocation failed: {message}")]
    AllocationError { message: String },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("WGPU error: {0}")]
    WgpuError(#[from] wgpu::Error),
}

impl BackendError {
    /// Create a new GPU error with message
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuError {
            message: message.into(),
        }
    }

    /// Create a new allocation error with message
    pub fn allocation_error<S: Into<String>>(message: S) -> Self {
        Self::AllocationError {
            message: message.into(),
        }
    }

    /// Create a new invalid operation error with message
    pub fn invalid_operation<S: Into<String>>(message: S) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }
}
