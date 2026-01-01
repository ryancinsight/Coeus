//! Error types for neural network operations.

/// Errors that can occur in neural network operations.
#[derive(thiserror::Error, Debug)]
pub enum NNError {
    #[error("Tensor operation failed: {source}")]
    TensorError {
        #[from]
        source: tensor::TensorError,
    },

    #[error("Storage operation failed: {source}")]
    StorageError {
        #[from]
        source: storage::StorageError,
    },

    #[error("Backend operation failed: {source}")]
    BackendError {
        #[from]
        source: backend::BackendError,
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

    #[error("Unsupported operation: {operation} - {reason}")]
    UnsupportedOperation { operation: String, reason: String },

    #[error("Invalid argument '{param}': {message}")]
    InvalidArgument { param: String, message: String },

    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    #[error("Resource allocation failed: {message}")]
    ResourceError { message: String },

    #[error("Not implemented: {operation}")]
    NotImplemented { operation: String },

    #[error("Serialization failed: {message}")]
    SerializationError { message: String },

    #[error("Invalid state: {message}")]
    InvalidState { message: String },

    #[error("Component not initialized: {component}")]
    NotInitialized { component: String },

    #[error("Execution error: {message}")]
    ExecutionError { message: String },

    #[error("I/O error: {error}")]
    IoError {
        #[from]
        error: std::io::Error,
    },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("JSON serialization error: {error}")]
    JsonError {
        #[from]
        error: serde_json::Error,
    },

    #[error("Autograd error: {error}")]
    AutogradError {
        #[from]
        error: autograd::AutogradError,
    },
}

/// Result type for neural network operations.
pub type Result<T> = std::result::Result<T, NNError>;
