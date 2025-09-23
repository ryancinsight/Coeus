//! Error types for the models crate.
//!
//! This module defines comprehensive error types for model loading, inference,
//! and quantization operations. All errors implement `std::error::Error` and
//! provide detailed context for debugging and user-facing error messages.

use std::io;
use std::string::FromUtf8Error;

/// Model-related errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// I/O errors during file operations
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// UTF-8 conversion errors
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] FromUtf8Error),

    /// Model format parsing errors
    #[error("Model format error: {message}")]
    Format { message: String },

    /// Quantization-related errors
    #[error("Quantization error: {message}")]
    Quantization { message: String },

    /// Inference engine errors
    #[error("Inference error: {message}")]
    Inference { message: String },

    /// Model loading errors
    #[error("Model loading error: {message}")]
    Loading { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Network/download errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Memory allocation errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {message}")]
    Unsupported { message: String },

    /// Architecture mismatch errors
    #[error("Architecture mismatch: expected {expected}, found {found}")]
    ArchitectureMismatch { expected: String, found: String },

    /// Version compatibility errors
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },

    /// Missing required components
    #[error("Missing component: {component}")]
    MissingComponent { component: String },

    /// Parameter validation errors
    #[error("Invalid parameter {parameter}: {message}")]
    InvalidParameter { parameter: String, message: String },

    /// Shape mismatch errors
    #[error("Shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    /// Type conversion errors
    #[error("Type conversion error: {message}")]
    TypeConversion { message: String },

    /// GPU-related errors
    #[error("GPU error: {message}")]
    Gpu { message: String },

    /// Custom error with arbitrary message
    #[error("{0}")]
    Custom(String),
}

impl ModelError {
    /// Create a format error
    pub fn format(message: impl Into<String>) -> Self {
        Self::Format {
            message: message.into(),
        }
    }

    /// Create a quantization error
    pub fn quantization(message: impl Into<String>) -> Self {
        Self::Quantization {
            message: message.into(),
        }
    }

    /// Create an inference error
    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }

    /// Create a loading error
    pub fn loading(message: impl Into<String>) -> Self {
        Self::Loading {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Create a type conversion error
    pub fn type_conversion(message: impl Into<String>) -> Self {
        Self::TypeConversion {
            message: message.into(),
        }
    }

    /// Create a GPU error
    pub fn gpu(message: impl Into<String>) -> Self {
        Self::Gpu {
            message: message.into(),
        }
    }

    /// Create a custom error
    pub fn custom(message: impl Into<String>) -> Self {
        Self::Custom(message.into())
    }

    /// Check if this is an I/O error
    pub fn is_io(&self) -> bool {
        matches!(self, Self::Io(_))
    }

    /// Check if this is a format error
    pub fn is_format(&self) -> bool {
        matches!(self, Self::Format { .. })
    }

    /// Check if this is a quantization error
    pub fn is_quantization(&self) -> bool {
        matches!(self, Self::Quantization { .. })
    }

    /// Check if this is an inference error
    pub fn is_inference(&self) -> bool {
        matches!(self, Self::Inference { .. })
    }

    /// Check if this is a network error
    pub fn is_network(&self) -> bool {
        matches!(self, Self::Network { .. })
    }

    /// Get the error message
    pub fn message(&self) -> String {
        match self {
            Self::Io(e) => e.kind().to_string(),
            Self::Utf8(e) => e.to_string(),
            Self::Format { message } => message.clone(),
            Self::Quantization { message } => message.clone(),
            Self::Inference { message } => message.clone(),
            Self::Loading { message } => message.clone(),
            Self::Config { message } => message.clone(),
            Self::Network { message } => message.clone(),
            Self::Serialization { message } => message.clone(),
            Self::Memory { message } => message.clone(),
            Self::Validation { message } => message.clone(),
            Self::Unsupported { message } => message.clone(),
            Self::ArchitectureMismatch { .. } => "architecture mismatch".to_string(),
            Self::VersionMismatch { .. } => "version mismatch".to_string(),
            Self::MissingComponent { .. } => "missing component".to_string(),
            Self::InvalidParameter { .. } => "invalid parameter".to_string(),
            Self::ShapeMismatch { .. } => "shape mismatch".to_string(),
            Self::TypeConversion { message } => message.clone(),
            Self::Gpu { message } => message.clone(),
            Self::Custom(message) => message.clone(),
        }
    }
}

/// Result type alias for model operations
pub type ModelResult<T> = Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let io_err = ModelError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        assert!(io_err.is_io());
        assert_eq!(io_err.message(), "entity not found");

        let format_err = ModelError::format("invalid magic number");
        assert!(format_err.is_format());
        assert_eq!(format_err.message(), "invalid magic number");

        let quant_err = ModelError::quantization("unsupported quantization scheme");
        assert!(quant_err.is_quantization());
        assert_eq!(quant_err.message(), "unsupported quantization scheme");

        let inference_err = ModelError::inference("model not loaded");
        assert!(inference_err.is_inference());
        assert_eq!(inference_err.message(), "model not loaded");

        let network_err = ModelError::network("connection timeout");
        assert!(network_err.is_network());
        assert_eq!(network_err.message(), "connection timeout");

        let custom_err = ModelError::custom("something went wrong");
        assert_eq!(custom_err.message(), "something went wrong");
    }

    #[test]
    fn test_error_display() {
        let err = ModelError::format("test message");
        assert_eq!(err.to_string(), "Model format error: test message");

        let err = ModelError::quantization("quantization failed");
        assert_eq!(err.to_string(), "Quantization error: quantization failed");

        let err = ModelError::ArchitectureMismatch {
            expected: "llama".to_string(),
            found: "gpt2".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Architecture mismatch: expected llama, found gpt2"
        );
    }

    #[test]
    fn test_architecture_mismatch() {
        let err = ModelError::ArchitectureMismatch {
            expected: "llama".to_string(),
            found: "gpt2".to_string(),
        };
        assert_eq!(err.message(), "architecture mismatch");
        assert_eq!(
            err.to_string(),
            "Architecture mismatch: expected llama, found gpt2"
        );
    }

    #[test]
    fn test_version_mismatch() {
        let err = ModelError::VersionMismatch {
            expected: "v1.0".to_string(),
            found: "v2.0".to_string(),
        };
        assert_eq!(err.message(), "version mismatch");
        assert_eq!(
            err.to_string(),
            "Version mismatch: expected v1.0, found v2.0"
        );
    }
}
