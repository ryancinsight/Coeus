//! Error types for the optimization crate

use thiserror::Error;

/// Errors that can occur during optimization operations
#[derive(Debug, Error)]
pub enum OptimError {
    #[error("Gradient not available for parameter")]
    GradientNotAvailable,

    #[error("Invalid optimizer state: {message}")]
    InvalidState { message: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Parameter not found: {name}")]
    ParameterNotFound { name: String },

    #[error("Learning rate must be positive, got {lr}")]
    InvalidLearningRate { lr: f32 },

    #[error("Weight decay must be non-negative, got {wd}")]
    InvalidWeightDecay { wd: f32 },
}
