//! Error types for the utils crate

use coeus_tensor;

/// Errors that can occur in data loading operations
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// Index out of bounds for dataset access
    #[error("Index {index} out of bounds for dataset with length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    /// Invalid batch size specification
    #[error("Invalid batch size: {batch_size}. Must be > 0")]
    InvalidBatchSize { batch_size: usize },

    /// Empty dataset error
    #[error("Dataset is empty")]
    EmptyDataset,

    /// I/O error during data loading
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    /// Invalid dataset configuration
    #[error("Invalid dataset configuration: {message}")]
    InvalidConfiguration { message: String },

    /// Data format error
    #[error("Data format error: {message}")]
    FormatError { message: String },

    /// Tensor-related error
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Tensor error from tensor crate
    #[error("Tensor operation error: {0}")]
    TensorOpError(#[from] coeus_tensor::TensorError),
}

impl DataError {
    /// Create a new index out of bounds error
    pub fn index_out_of_bounds(index: usize, len: usize) -> Self {
        Self::IndexOutOfBounds { index, len }
    }

    /// Create a new invalid batch size error
    pub fn invalid_batch_size(batch_size: usize) -> Self {
        Self::InvalidBatchSize { batch_size }
    }

    /// Create a new invalid configuration error
    pub fn invalid_configuration<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create a new format error
    pub fn format_error<S: Into<String>>(message: S) -> Self {
        Self::FormatError {
            message: message.into(),
        }
    }

    /// Create a new tensor error
    pub fn tensor_error<S: Into<String>>(message: S) -> Self {
        Self::TensorError(message.into())
    }
}

/// Result type alias for data operations
pub type Result<T> = std::result::Result<T, DataError>;
