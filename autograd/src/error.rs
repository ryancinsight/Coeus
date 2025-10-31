//! Error types for the autograd system

use tensor::TensorError;
use thiserror::Error;

/// Result type alias for autograd operations
pub type Result<T> = std::result::Result<T, AutogradError>;

/// Errors that can occur during automatic differentiation operations
#[derive(Error, Debug)]
pub enum AutogradError {
    /// Attempted to access gradient of a variable that doesn't require gradients
    #[error("Variable does not require gradients")]
    NoGradient,

    /// Attempted to compute gradients for a tensor that is not part of any computation graph
    #[error("Tensor is not part of any computation graph")]
    NotInGraph,

    /// Memory allocation failed during gradient computation
    #[error("Memory allocation failed: {message}")]
    MemoryAllocation {
        /// Additional context about the allocation failure
        message: String,
    },

    /// Invalid operation attempted during gradient computation
    #[error("Invalid gradient operation: {operation}")]
    InvalidOperation {
        /// Description of the invalid operation
        operation: String,
    },

    /// Invalid input provided to autograd operation
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Description of the invalid input
        message: String,
    },

    /// Error in computation graph operations
    #[error("Computation graph error: {0}")]
    GraphError(String),

    /// Gradient computation failed due to numerical issues
    #[error("Numerical error in gradient computation: {details}")]
    NumericalError {
        /// Details about the numerical issue
        details: String,
    },

    /// Cycle detected in computation graph
    #[error("Computation graph contains a cycle")]
    GraphCycle,

    /// Operation not supported for automatic differentiation
    #[error("Operation '{operation}' is not differentiable")]
    NonDifferentiableOperation {
        /// Name of the non-differentiable operation
        operation: String,
    },

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {operation}")]
    NotImplemented {
        /// Description of the unimplemented feature
        operation: String,
    },

    /// Error occurred during gradient accumulation
    #[error("Gradient accumulation failed: {message}")]
    GradientError {
        /// Description of the gradient accumulation failure
        message: String,
    },

    /// Error during gradient computation in backward pass
    #[error("Gradient computation failed in {operation}: {source}")]
    GradientComputationError {
        /// The operation that failed
        operation: String,
        /// The underlying error that occurred
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Tensor operation error
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    /// Storage operation error
    #[error("Storage error: {0}")]
    StorageError(#[from] storage::StorageError),
}
