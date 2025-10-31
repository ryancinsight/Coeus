//! Error types for the optimization crate

use thiserror::Error;

/// Errors that can occur during optimization operations
#[derive(Debug, Error)]
pub enum OptimError {
    #[error("Gradient not available for parameter")]
    GradientNotAvailable,

    #[error(
        "Invalid optimizer state for parameter '{param_name}': missing state key '{state_key}'"
    )]
    InvalidState {
        param_name: String,
        state_key: String,
    },

    #[error("Shape mismatch for parameter '{param_name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        param_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Parameter not found: {name}")]
    ParameterNotFound { name: String },

    #[error("Learning rate must be positive, got {lr}")]
    InvalidLearningRate { lr: f32 },

    #[error("Weight decay must be non-negative, got {wd}")]
    InvalidWeightDecay { wd: f32 },

    #[error("Invalid hyperparameter '{param}': {value} - {reason}")]
    InvalidHyperparameter {
        param: String,
        value: f64,
        reason: String,
    },

    #[error("Invalid parameter '{param}': {value} - {reason}")]
    InvalidParameter {
        param: String,
        value: String,
        reason: String,
    },

    #[error("Tensor operation failed: {source}")]
    TensorError {
        #[from]
        source: tensor::TensorError,
    },

    #[error("Backend operation failed: {message}")]
    BackendError { message: String },
}
