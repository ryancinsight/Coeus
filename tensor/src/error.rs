//! Error types for tensor operations

use core::fmt;

/// Errors that can occur during tensor operations
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorError {
    /// Error from storage layer
    StorageError(storage::StorageError),

    /// Shape mismatch in operations
    ShapeMismatch {
        /// Expected shape
        expected: std::vec::Vec<usize>,
        /// Actual shape
        actual: std::vec::Vec<usize>,
        /// Operation that failed
        operation: &'static str,
    },

    /// Shape error in reshape/transpose operations
    ShapeError {
        /// Expected value (if applicable)
        expected: usize,
        /// Actual value (if applicable)
        actual: usize,
        /// Error message
        message: std::string::String,
    },

    /// Autograd-related error
    AutogradError(std::string::String),

    /// Invalid operation for given dtype
    InvalidOperation {
        /// Operation name
        operation: &'static str,
        /// Dtype that doesn't support it
        dtype: dtype::Dtype,
        /// Reason for failure
        reason: &'static str,
    },

    /// Error from backend layer
    BackendError(std::string::String),

    /// Unsupported operation for storage type
    UnsupportedOperation {
        /// Operation name
        operation: std::string::String,
        /// Storage type
        storage_type: std::string::String,
    },

    /// Broadcasting error between tensors
    BroadcastError {
        /// Left-hand side shape
        lhs_shape: std::vec::Vec<usize>,
        /// Right-hand side shape
        rhs_shape: std::vec::Vec<usize>,
    },

    /// Operation on empty tensor
    EmptyTensor,

    /// Invalid dimension index
    InvalidDimension {
        /// Dimension index attempted
        dim: usize,
        /// Number of dimensions in tensor
        ndim: usize,
    },

    /// Invalid range specification
    InvalidRange {
        /// Start index
        start: usize,
        /// End index
        end: usize,
        /// Size of dimension
        size: usize,
    },

    /// Feature not implemented
    NotImplemented(std::string::String),

    /// Invalid input provided
    InvalidInput {
        /// Error message
        message: std::string::String,
    },
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageError(e) => write!(f, "Storage error: {e}"),
            Self::ShapeMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Shape mismatch in {operation}: expected {expected:?}, got {actual:?}"
                )
            }
            Self::ShapeError { message, .. } => {
                write!(f, "Shape error: {message}")
            }
            Self::AutogradError(msg) => {
                write!(f, "Autograd error: {msg}")
            }
            Self::InvalidOperation {
                operation,
                dtype,
                reason,
            } => {
                write!(f, "Invalid {operation} for {dtype}: {reason}")
            }
            Self::BackendError(msg) => {
                write!(f, "Backend error: {msg}")
            }
            Self::BroadcastError {
                lhs_shape: _,
                rhs_shape: _,
            } => {
                write!(f, "Incompatible shapes for broadcasting")
            }
            Self::UnsupportedOperation {
                operation,
                storage_type,
            } => {
                write!(
                    f,
                    "Unsupported {operation} operation for {storage_type} storage type"
                )
            }
            Self::EmptyTensor => {
                write!(f, "Operation on empty tensor")
            }
            Self::InvalidDimension { dim, ndim } => {
                write!(f, "Invalid dimension {dim} for tensor with {ndim} dimensions")
            }
            Self::InvalidRange { start, end, size } => {
                write!(f, "Invalid range [{start}, {end}) for dimension of size {size}")
            }
            Self::InvalidInput { message } => {
                write!(f, "Invalid input: {message}")
            }
            Self::NotImplemented(msg) => {
                write!(f, "Not implemented: {msg}")
            }
        }
    }
}

impl From<storage::StorageError> for TensorError {
    fn from(error: storage::StorageError) -> Self {
        Self::StorageError(error)
    }
}

impl From<backend::BackendError> for TensorError {
    fn from(error: backend::BackendError) -> Self {
        match error {
            backend::BackendError::UnsupportedOperation { operation, backend } => {
                Self::BackendError(std::format!(
                    "Unsupported {operation} operation for {backend} backend"
                ))
            }
            backend::BackendError::InvalidInput(msg) => {
                Self::BackendError(std::format!("Invalid input: {msg}"))
            }
            backend::BackendError::StorageError { source } => Self::StorageError(source),
            backend::BackendError::GpuError(msg) => {
                Self::BackendError(std::format!("GPU error: {msg}"))
            }
        }
    }
}
