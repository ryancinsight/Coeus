//! Error types for neural network operations.

/// Errors that can occur in neural network operations.
#[derive(thiserror::Error, Debug)]
pub enum NNError {
    #[error("Tensor operation failed: {source}")]
    TensorError {
        #[from]
        source: coeus_tensor::TensorError,
    },

    #[cfg(feature = "autograd")]
    #[error("Autograd operation failed: {source}")]
    AutogradError {
        #[from]
        source: coeus_autograd::AutogradError,
    },

    #[cfg(not(feature = "autograd"))]
    #[error("Autograd operation failed: {source}")]
    AutogradError {
        #[from]
        source: crate::autograd_stub::AutogradError,
    },

    #[error("Storage operation failed: {source}")]
    StorageError {
        #[from]
        source: coeus_storage::StorageError,
    },

    #[error("Invalid parameter shape: expected {expected:?}, got {actual:?}")]
    InvalidParameterShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Module {name} not found")]
    ModuleNotFound { name: String },

    #[error("Parameter {name} not found")]
    ParameterNotFound { name: String },

    #[error("Training error: {message}")]
    TrainingError { message: String },

    #[error("Shape mismatch in {operation}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        operation: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid module configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Serialization failed: {message}")]
    SerializationError { message: String },
}

/// Result type for neural network operations.
pub type Result<T> = std::result::Result<T, NNError>;
