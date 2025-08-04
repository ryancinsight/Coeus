//! Error handling for RustLLM Core.
//!
//! This module provides a zero-dependency error handling system that follows
//! Rust best practices while maintaining simplicity and performance.

use core::fmt;

/// The main error type for RustLLM Core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Plugin-related errors.
    Plugin(PluginError),
    
    /// Tokenizer-related errors.
    Tokenizer(TokenizerError),
    
    /// Model-related errors.
    Model(ModelError),
    
    /// I/O errors (when std feature is enabled).
    #[cfg(feature = "std")]
    Io(String),
    
    /// Configuration errors.
    Config(String),
    
    /// Invalid input provided.
    InvalidInput(String),
    
    /// Operation not supported.
    NotSupported(String),
    
    /// Resource not found.
    NotFound(String),
    
    /// Generic error with a message.
    Other(String),
}

/// Plugin-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    /// Plugin not found.
    NotFound(String),
    
    /// Plugin initialization failed.
    InitializationFailed(String),
    
    /// Plugin version mismatch.
    VersionMismatch {
        /// Expected version.
        expected: String,
        /// Actual version.
        actual: String,
    },
    
    /// Plugin already loaded.
    AlreadyLoaded(String),
    
    /// Plugin dependency missing.
    DependencyMissing(String),
}

/// Tokenizer-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Invalid token encountered.
    InvalidToken(String),
    
    /// Vocabulary not loaded.
    VocabularyNotLoaded,
    
    /// Encoding failed.
    EncodingFailed(String),
    
    /// Decoding failed.
    DecodingFailed(String),
    
    /// Token limit exceeded.
    TokenLimitExceeded {
        /// Maximum allowed tokens.
        limit: usize,
        /// Actual token count.
        actual: usize,
    },
}

/// Model-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model not initialized.
    NotInitialized,
    
    /// Invalid model configuration.
    InvalidConfig(String),
    
    /// Inference failed.
    InferenceFailed(String),
    
    /// Model format not supported.
    UnsupportedFormat(String),
    
    /// Shape mismatch in tensors.
    ShapeMismatch {
        /// Expected shape.
        expected: String,
        /// Actual shape.
        actual: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plugin(e) => write!(f, "Plugin error: {e}"),
            Self::Tokenizer(e) => write!(f, "Tokenizer error: {e}"),
            Self::Model(e) => write!(f, "Model error: {e}"),
            #[cfg(feature = "std")]
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Config(msg) => write!(f, "Configuration error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            Self::NotSupported(msg) => write!(f, "Not supported: {msg}"),
            Self::NotFound(msg) => write!(f, "Not found: {msg}"),
            Self::Other(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "Plugin '{name}' not found"),
            Self::InitializationFailed(msg) => write!(f, "Plugin initialization failed: {msg}"),
            Self::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
            Self::AlreadyLoaded(name) => write!(f, "Plugin '{name}' already loaded"),
            Self::DependencyMissing(dep) => write!(f, "Missing dependency: {dep}"),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken(token) => write!(f, "Invalid token: {token}"),
            Self::VocabularyNotLoaded => write!(f, "Vocabulary not loaded"),
            Self::EncodingFailed(msg) => write!(f, "Encoding failed: {msg}"),
            Self::DecodingFailed(msg) => write!(f, "Decoding failed: {msg}"),
            Self::TokenLimitExceeded { limit, actual } => {
                write!(f, "Token limit exceeded: {actual} > {limit}")
            }
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "Model not initialized"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl std::error::Error for PluginError {}

#[cfg(feature = "std")]
impl std::error::Error for TokenizerError {}

#[cfg(feature = "std")]
impl std::error::Error for ModelError {}

/// A specialized Result type for RustLLM Core operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Extension trait for converting other error types to our Error type.
pub trait ErrorContext<T> {
    /// Provide context for an error.
    fn context(self, msg: &str) -> Result<T>;
    
    /// Provide context with a closure.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for core::result::Result<T, E>
where
    E: fmt::Display,
{
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| Error::Other(format!("{msg}: {e}")))
    }
    
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| Error::Other(format!("{}: {e}", f())))
    }
}

impl<T> ErrorContext<T> for Option<T> {
    fn context(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| Error::NotFound(msg.to_string()))
    }
    
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.ok_or_else(|| Error::NotFound(f()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let error = Error::Plugin(PluginError::NotFound("test".to_string()));
        assert_eq!(error.to_string(), "Plugin error: Plugin 'test' not found");
    }
    
    #[test]
    fn test_error_context() {
        let result: core::result::Result<(), &str> = Err("failed");
        let error = result.context("Operation failed").unwrap_err();
        assert_eq!(error.to_string(), "Error: Operation failed: failed");
    }
    
    #[test]
    fn test_option_context() {
        let option: Option<()> = None;
        let error = option.context("Value not found").unwrap_err();
        assert_eq!(error.to_string(), "Not found: Value not found");
    }
}