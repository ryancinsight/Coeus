//! Error types for tokenizer operations.

/// Errors that can occur during tokenization operations.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// Invalid UTF-8 encoding in input text.
    #[error("Invalid UTF-8 encoding: {0}")]
    InvalidUtf8(String),

    /// Unknown token not found in vocabulary.
    #[error("Unknown token: {0}")]
    UnknownToken(String),

    /// Invalid token ID not found in vocabulary.
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),

    /// Vocabulary serialization/deserialization error.
    #[error("Vocabulary error: {0}")]
    VocabularyError(String),

    /// Unicode normalization error.
    #[error("Unicode normalization error: {0}")]
    UnicodeError(String),

    /// Encoding/decoding operation failed.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// I/O error during model loading/saving.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Regex compilation error.
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),

    /// Custom error with context.
    #[error("{context}: {source}")]
    Custom {
        /// Error context
        context: String,
        /// Source error
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl TokenizerError {
    /// Create a custom error with context.
    pub fn custom<E>(context: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Custom {
            context: context.into(),
            source: Box::new(source),
        }
    }

    /// Create an encoding error.
    pub fn encoding(msg: impl Into<String>) -> Self {
        Self::EncodingError(msg.into())
    }

    /// Create a vocabulary error.
    pub fn vocabulary(msg: impl Into<String>) -> Self {
        Self::VocabularyError(msg.into())
    }

    /// Create a Unicode error.
    pub fn unicode(msg: impl Into<String>) -> Self {
        Self::UnicodeError(msg.into())
    }
}

/// Result type alias for tokenizer operations.
pub type Result<T> = std::result::Result<T, TokenizerError>;
