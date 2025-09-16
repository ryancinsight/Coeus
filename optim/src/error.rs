//! Error types for the optimization crate

/// Errors that can occur during optimization operations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimError {
    /// Invalid parameter configuration
    InvalidParameter(String),
    /// Learning rate is invalid (negative, NaN, etc.)
    InvalidLearningRate(String),
    /// Parameter group index out of bounds
    ParameterGroupOutOfBounds(usize),
    /// Parameter has no gradient
    NoGradient(String),
    /// Tensor operation failed
    TensorError(String),
    /// Custom error message
    Other(String),
}

impl std::fmt::Display for OptimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            OptimError::InvalidLearningRate(msg) => write!(f, "Invalid learning rate: {}", msg),
            OptimError::ParameterGroupOutOfBounds(idx) => {
                write!(f, "Parameter group index {} out of bounds", idx)
            }
            OptimError::NoGradient(param) => write!(f, "Parameter {} has no gradient", param),
            OptimError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            OptimError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for OptimError {}

impl From<coeus_tensor::TensorError> for OptimError {
    fn from(err: coeus_tensor::TensorError) -> Self {
        OptimError::TensorError(err.to_string())
    }
}

impl From<String> for OptimError {
    fn from(msg: String) -> Self {
        OptimError::Other(msg)
    }
}

impl From<&str> for OptimError {
    fn from(msg: &str) -> Self {
        OptimError::Other(msg.to_string())
    }
}

impl From<anyhow::Error> for OptimError {
    fn from(err: anyhow::Error) -> Self {
        OptimError::Other(err.to_string())
    }
}
