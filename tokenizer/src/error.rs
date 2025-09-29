//! Error types and handling for the tokenizer crate

use thiserror::Error;

/// Result type alias for tokenizer operations
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Comprehensive error types for tokenizer operations
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum TokenizerError {
    /// Invalid input text or token sequence
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message describing the invalid input
        message: String,
    },

    /// Vocabulary-related errors
    #[error("Vocabulary error: {message}")]
    VocabularyError {
        /// Error message describing the vocabulary issue
        message: String,
    },

    /// Encoding/decoding operation failed
    #[error("Encoding error: {message}")]
    EncodingError {
        /// Error message describing the encoding issue
        message: String,
    },

    /// Model or tokenizer configuration error
    #[error("Model error: {message}")]
    ModelError {
        /// Error message describing the model configuration issue
        message: String,
    },

    /// Special token handling error
    #[error("Special token error: {message}")]
    SpecialTokenError {
        /// Error message describing the special token issue
        message: String,
    },

    /// I/O error during vocabulary loading/saving
    #[error("I/O error: {message}")]
    IoError {
        /// Error message describing the I/O issue
        message: String,
    },

    /// Sequence length exceeds maximum allowed
    #[error("Sequence too long: {length} tokens (max: {max_length})")]
    SequenceTooLong {
        /// Actual sequence length
        length: usize,
        /// Maximum allowed length
        max_length: usize,
    },

    /// Unknown token encountered during encoding
    #[error("Unknown token: {token}")]
    UnknownToken {
        /// The unknown token that was encountered
        token: String,
    },

    /// Invalid token ID
    #[error("Invalid token ID: {token_id}")]
    InvalidTokenId {
        /// The invalid token ID
        token_id: usize,
    },

    /// BPE merge operation failed
    #[error("BPE merge error: {message}")]
    BpeMergeError {
        /// Error message describing the BPE merge issue
        message: String,
    },

    /// Batch processing error
    #[error("Batch processing error: {message}")]
    BatchError {
        /// Error message describing the batch processing issue
        message: String,
    },

    /// Memory allocation or capacity error
    #[error("Memory error: {message}")]
    MemoryError {
        /// Error message describing the memory issue
        message: String,
    },

    /// Unsupported operation for the current model
    #[error("Unsupported operation: {operation} not supported for model {model}")]
    UnsupportedOperation {
        /// The operation that is not supported
        operation: String,
        /// The model for which the operation is not supported
        model: String,
    },

    /// Configuration validation error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message describing the configuration issue
        message: String,
    },
}

impl TokenizerError {
    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a vocabulary error
    pub fn vocabulary_error<S: Into<String>>(message: S) -> Self {
        Self::VocabularyError {
            message: message.into(),
        }
    }

    /// Create an encoding error
    pub fn encoding_error<S: Into<String>>(message: S) -> Self {
        Self::EncodingError {
            message: message.into(),
        }
    }

    /// Create a model error
    pub fn model_error<S: Into<String>>(message: S) -> Self {
        Self::ModelError {
            message: message.into(),
        }
    }

    /// Create a special token error
    pub fn special_token_error<S: Into<String>>(message: S) -> Self {
        Self::SpecialTokenError {
            message: message.into(),
        }
    }

    /// Create an I/O error
    pub fn io_error<S: Into<String>>(message: S) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Create a sequence too long error
    #[must_use]
    pub const fn sequence_too_long(length: usize, max_length: usize) -> Self {
        Self::SequenceTooLong { length, max_length }
    }

    /// Create an unknown token error
    pub fn unknown_token<S: Into<String>>(token: S) -> Self {
        Self::UnknownToken {
            token: token.into(),
        }
    }

    /// Create an invalid token ID error
    #[must_use]
    pub const fn invalid_token_id(token_id: usize) -> Self {
        Self::InvalidTokenId { token_id }
    }

    /// Create a BPE merge error
    pub fn bpe_merge_error<S: Into<String>>(message: S) -> Self {
        Self::BpeMergeError {
            message: message.into(),
        }
    }

    /// Create a batch processing error
    pub fn batch_error<S: Into<String>>(message: S) -> Self {
        Self::BatchError {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory_error<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation<S: Into<String>, T: Into<String>>(operation: S, model: T) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            model: model.into(),
        }
    }

    /// Create a configuration error
    pub fn config_error<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Check if this is a recoverable error
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput { .. }
                | Self::SequenceTooLong { .. }
                | Self::UnknownToken { .. }
                | Self::InvalidTokenId { .. }
                | Self::BatchError { .. }
        )
    }

    /// Check if this is a fatal configuration error
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ConfigError { .. }
                | Self::ModelError { .. }
                | Self::VocabularyError { .. }
                | Self::UnsupportedOperation { .. }
        )
    }

    /// Get the error category for logging/metrics
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::InvalidInput { .. } => "invalid_input",
            Self::VocabularyError { .. } => "vocabulary",
            Self::EncodingError { .. } => "encoding",
            Self::ModelError { .. } => "model",
            Self::SpecialTokenError { .. } => "special_tokens",
            Self::IoError { .. } => "io",
            Self::SequenceTooLong { .. } => "sequence_length",
            Self::UnknownToken { .. } => "unknown_token",
            Self::InvalidTokenId { .. } => "invalid_token_id",
            Self::BpeMergeError { .. } => "bpe_merge",
            Self::BatchError { .. } => "batch",
            Self::MemoryError { .. } => "memory",
            Self::UnsupportedOperation { .. } => "unsupported_operation",
            Self::ConfigError { .. } => "configuration",
        }
    }
}

impl From<coeus_tensor::TensorError> for TokenizerError {
    fn from(error: coeus_tensor::TensorError) -> Self {
        Self::encoding_error(format!("Tensor operation failed: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = TokenizerError::invalid_input("test message");
        assert_eq!(err.category(), "invalid_input");
        assert!(!err.is_fatal());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_fatal_errors() {
        let err = TokenizerError::config_error("test");
        assert!(err.is_fatal());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = TokenizerError::sequence_too_long(1000, 512);
        let msg = format!("{err}");
        assert!(msg.contains("1000"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(
            TokenizerError::invalid_input("").category(),
            "invalid_input"
        );
        assert_eq!(
            TokenizerError::vocabulary_error("").category(),
            "vocabulary"
        );
        assert_eq!(TokenizerError::encoding_error("").category(), "encoding");
        assert_eq!(TokenizerError::model_error("").category(), "model");
        assert_eq!(
            TokenizerError::special_token_error("").category(),
            "special_tokens"
        );
        assert_eq!(TokenizerError::io_error("").category(), "io");
        assert_eq!(
            TokenizerError::sequence_too_long(0, 0).category(),
            "sequence_length"
        );
        assert_eq!(
            TokenizerError::unknown_token("").category(),
            "unknown_token"
        );
        assert_eq!(
            TokenizerError::invalid_token_id(0).category(),
            "invalid_token_id"
        );
        assert_eq!(TokenizerError::bpe_merge_error("").category(), "bpe_merge");
        assert_eq!(TokenizerError::batch_error("").category(), "batch");
        assert_eq!(TokenizerError::memory_error("").category(), "memory");
        assert_eq!(
            TokenizerError::unsupported_operation("", "").category(),
            "unsupported_operation"
        );
        assert_eq!(TokenizerError::config_error("").category(), "configuration");
    }

    #[test]
    fn test_error_recovery() {
        // Test recoverable errors
        assert!(TokenizerError::invalid_input("").is_recoverable());
        assert!(TokenizerError::unknown_token("").is_recoverable());
        assert!(TokenizerError::invalid_token_id(0).is_recoverable());
        assert!(TokenizerError::batch_error("").is_recoverable());
        assert!(TokenizerError::sequence_too_long(100, 50).is_recoverable());

        // Test fatal errors
        assert!(TokenizerError::config_error("").is_fatal());
        assert!(TokenizerError::model_error("").is_fatal());
        assert!(TokenizerError::vocabulary_error("").is_fatal());
        assert!(TokenizerError::unsupported_operation("", "").is_fatal());
    }

    #[test]
    fn test_error_display_formatting() {
        // Test sequence too long error formatting
        let err = TokenizerError::sequence_too_long(1000, 512);
        let msg = format!("{err}");
        assert!(msg.contains("1000"));
        assert!(msg.contains("512"));
        assert!(msg.contains("Sequence too long"));

        // Test invalid token ID error formatting
        let err = TokenizerError::invalid_token_id(999);
        let msg = format!("{err}");
        assert!(msg.contains("999"));
        assert!(msg.contains("Invalid token ID"));

        // Test unsupported operation error formatting
        let err = TokenizerError::unsupported_operation("encode", "gpt2");
        let msg = format!("{err}");
        assert!(msg.contains("encode"));
        assert!(msg.contains("gpt2"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_error_edge_cases() {
        // Test with very long error messages
        let long_message = "x".repeat(10000);
        let err = TokenizerError::invalid_input(&long_message);
        let msg = format!("{err}");
        assert!(msg.len() > 1000); // Should handle long messages

        // Test with unicode in error messages
        let unicode_message = "Error with unicode: 🚀 你好 🌟";
        let err = TokenizerError::vocabulary_error(unicode_message);
        let msg = format!("{err}");
        assert!(msg.contains("🚀"));
        assert!(msg.contains("你好"));

        // Test with empty error messages
        let err = TokenizerError::encoding_error("");
        let msg = format!("{err}");
        assert!(!msg.is_empty()); // Should still format properly

        // Test special token error with various token formats
        let tokens = vec!["[CLS]", "[SEP]", "[MASK]", "<|endoftext|>", "[BOS]"];
        for token in tokens {
            let err =
                TokenizerError::special_token_error(format!("Invalid special token: {token}"));
            let msg = format!("{err}");
            assert!(msg.contains(token));
        }
    }
}
