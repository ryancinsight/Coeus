//! Error handling for `RustLLM` Core.
//!
//! This module provides a zero-dependency error handling system that follows
//! Rust best practices while maintaining simplicity and performance.
//!
//! The error system follows these principles:
//! - **Composability**: Errors can be easily combined and wrapped
//! - **Context**: Errors carry sufficient context for debugging
//! - **Zero-cost**: No runtime overhead for the happy path
//! - **Type-safe**: Leverages Rust's type system for safety

use core::fmt;

/// The main result type for `RustLLM` Core.
pub type Result<T> = core::result::Result<T, Error>;

/// The main error type for `RustLLM` Core.
/// 
/// This type follows the principle of exhaustive error handling,
/// where each variant represents a specific error category.
#[derive(Debug, Clone)]
pub enum Error {
    /// Plugin-related errors.
    Plugin(PluginError),
    
    /// Tokenizer-related errors.
    Tokenizer(TokenizerError),
    
    /// Model-related errors.
    Model(ModelError),
    
    /// I/O errors (when std feature is enabled).
    #[cfg(feature = "std")]
    Io(IoError),
    
    /// Configuration errors.
    Config(ConfigError),
    
    /// Validation errors.
    Validation(ValidationError),
    
    /// Resource errors.
    Resource(ResourceError),
    
    /// Processing errors.
    Processing(ProcessingError),
    
    /// Composition of multiple errors.
    Multiple(Vec<Error>),
}

/// Error context that can be attached to any error.
/// 
/// This follows the principle of rich error information
/// without sacrificing performance.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation that was being performed.
    pub operation: &'static str,
    
    /// Additional context information.
    pub context: String,
    
    /// Source location (file, line, column).
    pub location: Option<(&'static str, u32, u32)>,
}

impl ErrorContext {
    /// Creates a new error context.
    pub const fn new(operation: &'static str) -> Self {
        Self {
            operation,
            context: String::new(),
            location: None,
        }
    }
    
    /// Adds context information.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }
    
    /// Adds source location.
    #[must_use]
    pub const fn at_location(mut self, file: &'static str, line: u32, column: u32) -> Self {
        self.location = Some((file, line, column));
        self
    }
}

/// Trait for adding context to errors.
/// 
/// This trait follows the decorator pattern for error enhancement.
pub trait ErrorExt: Sized {
    /// Adds context to the error.
    fn context(self, ctx: ErrorContext) -> ContextualError<Self>;
    
    /// Adds a simple string context.
    fn with_context(self, context: impl Into<String>) -> ContextualError<Self>;
}

impl<E> ErrorExt for E {
    fn context(self, ctx: ErrorContext) -> ContextualError<Self> {
        ContextualError {
            error: self,
            context: ctx,
        }
    }
    
    fn with_context(self, context: impl Into<String>) -> ContextualError<Self> {
        self.context(ErrorContext::new("operation").with_context(context))
    }
}

/// An error with attached context.
#[derive(Debug, Clone)]
pub struct ContextualError<E> {
    /// The underlying error.
    pub error: E,
    
    /// The error context.
    pub context: ErrorContext,
}

impl<E: fmt::Display> fmt::Display for ContextualError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context.operation, self.error)?;
        if !self.context.context.is_empty() {
            write!(f, " ({})", self.context.context)?;
        }
        if let Some((file, line, col)) = self.context.location {
            write!(f, " at {file}:{line}:{col}")?;
        }
        Ok(())
    }
}

/// Plugin-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    /// Plugin not found.
    NotFound {
        /// Name of the plugin that was not found.
        name: String
    },

    /// Plugin initialization failed.
    InitializationFailed {
        /// Name of the plugin that failed to initialize.
        name: String,
        /// Reason for the initialization failure.
        reason: String
    },

    /// Plugin version mismatch.
    VersionMismatch {
        /// Name of the plugin with version mismatch.
        name: String,
        /// Expected version specification.
        expected: String,
        /// Actual version found.
        actual: String,
    },

    /// Plugin already loaded.
    AlreadyLoaded {
        /// Name of the plugin that is already loaded.
        name: String
    },

    /// Plugin dependency missing.
    DependencyMissing {
        /// Name of the plugin with missing dependency.
        plugin: String,
        /// Name of the missing dependency.
        dependency: String,
    },

    /// Plugin state error.
    InvalidState {
        /// Name of the plugin with invalid state.
        plugin: String,
        /// Expected state for the operation.
        expected: &'static str,
        /// Actual current state of the plugin.
        actual: &'static str,
    },
}

/// Tokenizer-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Invalid token encountered.
    InvalidToken {
        /// The invalid token string.
        token: String,
        /// Position in the input where the invalid token was found.
        position: usize,
    },

    /// Vocabulary not loaded.
    VocabularyNotLoaded,

    /// Encoding failed.
    EncodingFailed {
        /// Input text that failed to encode.
        input: String,
        /// Reason for the encoding failure.
        reason: String,
    },

    /// Decoding failed.
    DecodingFailed {
        /// Reason for the decoding failure.
        reason: String,
    },

    /// Token limit exceeded.
    TokenLimitExceeded {
        /// Maximum allowed number of tokens.
        limit: usize,
        /// Actual number of tokens encountered.
        actual: usize,
    },

    /// Unknown token ID.
    UnknownTokenId {
        /// The unknown token ID.
        id: u32
    },
}

/// Model-specific errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model not initialized.
    NotInitialized,
    
    /// Invalid model configuration.
    InvalidConfig {
        /// Name of the configuration field that is invalid.
        field: String,
        /// Reason why the configuration is invalid.
        reason: String,
    },

    /// Inference failed.
    InferenceFailed {
        /// Reason for the inference failure.
        reason: String
    },

    /// Model format not supported.
    UnsupportedFormat {
        /// The unsupported model format.
        format: String
    },

    /// Shape mismatch.
    ShapeMismatch {
        /// Expected tensor shape.
        expected: String,
        /// Actual tensor shape encountered.
        actual: String,
    },

    /// Parameter error.
    InvalidParameter {
        /// Name of the invalid parameter.
        name: String,
        /// Reason why the parameter is invalid.
        reason: String,
    },

    /// Training error.
    TrainingFailed {
        /// Reason for the training failure.
        reason: String
    },
}

/// I/O errors for std environments.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    /// File not found.
    FileNotFound {
        /// Path to the file that was not found.
        path: String
    },

    /// Permission denied.
    PermissionDenied {
        /// Path to the file with permission issues.
        path: String
    },

    /// Read error.
    ReadError {
        /// Path to the file that failed to read.
        path: String,
        /// Reason for the read failure.
        reason: String
    },

    /// Write error.
    WriteError {
        /// Path to the file that failed to write.
        path: String,
        /// Reason for the write failure.
        reason: String
    },

    /// Network error.
    NetworkError {
        /// Reason for the network error.
        reason: String
    },
}

/// Configuration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Missing required field.
    MissingField {
        /// Name of the missing required field.
        field: String
    },

    /// Invalid field value.
    InvalidValue {
        /// Name of the field with invalid value.
        field: String,
        /// The invalid value that was provided.
        value: String,
        /// Description of the expected value format.
        expected: String,
    },

    /// Type mismatch.
    TypeMismatch {
        /// Name of the field with type mismatch.
        field: String,
        /// Expected type for the field.
        expected: &'static str,
        /// Actual type that was provided.
        actual: &'static str,
    },
    
    /// Validation failed.
    ValidationFailed {
        /// Name of the field that failed validation.
        field: String,
        /// Reason for the validation failure.
        reason: String,
    },
}

/// Validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Range validation failed.
    OutOfRange {
        /// The value that is out of range.
        value: String,
        /// Minimum allowed value, if any.
        min: Option<String>,
        /// Maximum allowed value, if any.
        max: Option<String>,
    },

    /// Pattern validation failed.
    PatternMismatch {
        /// The value that doesn't match the pattern.
        value: String,
        /// The expected pattern.
        pattern: String,
    },

    /// Constraint violation.
    ConstraintViolation {
        /// Description of the violated constraint.
        constraint: String,
        /// The value that violates the constraint.
        value: String,
    },

    /// Invalid state.
    InvalidState {
        /// Current state description.
        current: String,
        /// Expected state description.
        expected: String,
    },
}

/// Resource errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceError {
    /// Resource not found.
    NotFound {
        /// Name or identifier of the resource that was not found.
        resource: String
    },

    /// Resource already exists.
    AlreadyExists {
        /// Name or identifier of the resource that already exists.
        resource: String
    },

    /// Resource busy.
    Busy {
        /// Name or identifier of the resource that is busy.
        resource: String
    },

    /// Insufficient resources.
    Insufficient {
        /// Name or type of the insufficient resource.
        resource: String,
        /// Amount of resource currently available.
        available: usize,
        /// Amount of resource required for the operation.
        required: usize,
    },

    /// Resource limit exceeded.
    LimitExceeded {
        /// Name or type of the resource that exceeded its limit.
        resource: String,
        /// The limit that was exceeded.
        limit: usize,
    },
}

/// Processing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingError {
    /// Timeout occurred.
    Timeout {
        /// Description of the operation that timed out.
        operation: String,
        /// Timeout duration in milliseconds.
        timeout_ms: u64
    },

    /// Operation cancelled.
    Cancelled {
        /// Description of the operation that was cancelled.
        operation: String
    },

    /// Invalid input.
    InvalidInput {
        /// Reason why the input is invalid.
        reason: String
    },

    /// Unsupported operation.
    Unsupported {
        /// Description of the unsupported operation.
        operation: String
    },

    /// Internal error.
    Internal {
        /// Reason for the internal error.
        reason: String
    },
}

// ============================================================================
// Display implementations
// ============================================================================

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plugin(e) => write!(f, "Plugin error: {e}"),
            Self::Tokenizer(e) => write!(f, "Tokenizer error: {e}"),
            Self::Model(e) => write!(f, "Model error: {e}"),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Config(e) => write!(f, "Configuration error: {e}"),
            Self::Validation(e) => write!(f, "Validation error: {e}"),
            Self::Resource(e) => write!(f, "Resource error: {e}"),
            Self::Processing(e) => write!(f, "Processing error: {e}"),
            Self::Multiple(errors) => {
                write!(f, "Multiple errors occurred:")?;
                for (i, e) in errors.iter().enumerate() {
                    write!(f, "\n  {}. {e}", i + 1)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { name } => 
                write!(f, "Plugin '{name}' not found"),
            Self::InitializationFailed { name, reason } => 
                write!(f, "Plugin '{name}' initialization failed: {reason}"),
            Self::VersionMismatch { name, expected, actual } => 
                write!(f, "Plugin '{name}' version mismatch: expected {expected}, got {actual}"),
            Self::AlreadyLoaded { name } => 
                write!(f, "Plugin '{name}' is already loaded"),
            Self::DependencyMissing { plugin, dependency } => 
                write!(f, "Plugin '{plugin}' missing dependency '{dependency}'"),
            Self::InvalidState { plugin, expected, actual } => 
                write!(f, "Plugin '{plugin}' in invalid state: expected {expected}, got {actual}"),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken { token, position } => 
                write!(f, "Invalid token '{token}' at position {position}"),
            Self::VocabularyNotLoaded => 
                write!(f, "Vocabulary not loaded"),
            Self::EncodingFailed { input, reason } => 
                write!(f, "Failed to encode '{input}': {reason}"),
            Self::DecodingFailed { reason } => 
                write!(f, "Failed to decode: {reason}"),
            Self::TokenLimitExceeded { limit, actual } => 
                write!(f, "Token limit exceeded: {actual} > {limit}"),
            Self::UnknownTokenId { id } => 
                write!(f, "Unknown token ID: {id}"),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => 
                write!(f, "Model not initialized"),
            Self::InvalidConfig { field, reason } => 
                write!(f, "Invalid configuration field '{field}': {reason}"),
            Self::InferenceFailed { reason } => 
                write!(f, "Inference failed: {reason}"),
            Self::UnsupportedFormat { format } => 
                write!(f, "Unsupported model format: {format}"),
            Self::ShapeMismatch { expected, actual } => 
                write!(f, "Shape mismatch: expected {expected}, got {actual}"),
            Self::InvalidParameter { name, reason } => 
                write!(f, "Invalid parameter '{name}': {reason}"),
            Self::TrainingFailed { reason } => 
                write!(f, "Training failed: {reason}"),
        }
    }
}

#[cfg(feature = "std")]
impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound { path } => 
                write!(f, "File not found: {path}"),
            Self::PermissionDenied { path } => 
                write!(f, "Permission denied: {path}"),
            Self::ReadError { path, reason } => 
                write!(f, "Failed to read '{path}': {reason}"),
            Self::WriteError { path, reason } => 
                write!(f, "Failed to write '{path}': {reason}"),
            Self::NetworkError { reason } => 
                write!(f, "Network error: {reason}"),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField { field } => 
                write!(f, "Missing required field: {field}"),
            Self::InvalidValue { field, value, expected } => 
                write!(f, "Invalid value for field '{field}': '{value}', expected {expected}"),
            Self::TypeMismatch { field, expected, actual } => 
                write!(f, "Type mismatch for field '{field}': expected {expected}, got {actual}"),
            Self::ValidationFailed { field, reason } => 
                write!(f, "Validation failed for field '{field}': {reason}"),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { value, min, max } => {
                write!(f, "Value '{value}' out of range")?;
                match (min, max) {
                    (Some(min), Some(max)) => write!(f, " [{min}, {max}]"),
                    (Some(min), None) => write!(f, " [{min}+)"),
                    (None, Some(max)) => write!(f, " (-∞, {max}]"),
                    (None, None) => Ok(()),
                }
            }
            Self::PatternMismatch { value, pattern } => 
                write!(f, "Value '{value}' does not match pattern '{pattern}'"),
            Self::ConstraintViolation { constraint, value } => 
                write!(f, "Constraint '{constraint}' violated by value '{value}'"),
            Self::InvalidState { current, expected } => 
                write!(f, "Invalid state: current '{current}', expected '{expected}'"),
        }
    }
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { resource } => 
                write!(f, "Resource not found: {resource}"),
            Self::AlreadyExists { resource } => 
                write!(f, "Resource already exists: {resource}"),
            Self::Busy { resource } => 
                write!(f, "Resource busy: {resource}"),
            Self::Insufficient { resource, available, required } => 
                write!(f, "Insufficient {resource}: {available} available, {required} required"),
            Self::LimitExceeded { resource, limit } => 
                write!(f, "Resource limit exceeded for {resource}: {limit}"),
        }
    }
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout { operation, timeout_ms } => 
                write!(f, "Operation '{operation}' timed out after {timeout_ms}ms"),
            Self::Cancelled { operation } => 
                write!(f, "Operation '{operation}' was cancelled"),
            Self::InvalidInput { reason } => 
                write!(f, "Invalid input: {reason}"),
            Self::Unsupported { operation } => 
                write!(f, "Unsupported operation: {operation}"),
            Self::Internal { reason } => 
                write!(f, "Internal error: {reason}"),
        }
    }
}

// ============================================================================
// Error trait implementation
// ============================================================================

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl std::error::Error for PluginError {}

#[cfg(feature = "std")]
impl std::error::Error for TokenizerError {}

#[cfg(feature = "std")]
impl std::error::Error for ModelError {}

#[cfg(feature = "std")]
impl std::error::Error for IoError {}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

#[cfg(feature = "std")]
impl std::error::Error for ValidationError {}

#[cfg(feature = "std")]
impl std::error::Error for ResourceError {}

#[cfg(feature = "std")]
impl std::error::Error for ProcessingError {}

// ============================================================================
// Conversion implementations
// ============================================================================

impl From<PluginError> for Error {
    fn from(err: PluginError) -> Self {
        Self::Plugin(err)
    }
}

impl From<TokenizerError> for Error {
    fn from(err: TokenizerError) -> Self {
        Self::Tokenizer(err)
    }
}

impl From<ModelError> for Error {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

#[cfg(feature = "std")]
impl From<IoError> for Error {
    fn from(err: IoError) -> Self {
        Self::Io(err)
    }
}

impl From<ConfigError> for Error {
    fn from(err: ConfigError) -> Self {
        Self::Config(err)
    }
}

impl From<ValidationError> for Error {
    fn from(err: ValidationError) -> Self {
        Self::Validation(err)
    }
}

impl From<ResourceError> for Error {
    fn from(err: ResourceError) -> Self {
        Self::Resource(err)
    }
}

impl From<ProcessingError> for Error {
    fn from(err: ProcessingError) -> Self {
        Self::Processing(err)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Creates a plugin not found error.
pub fn plugin_not_found(name: impl Into<String>) -> Error {
    Error::Plugin(PluginError::NotFound { name: name.into() })
}

/// Creates a model not initialized error.
pub const fn model_not_initialized() -> Error {
    Error::Model(ModelError::NotInitialized)
}

/// Creates a validation out of range error.
pub fn out_of_range(
    value: impl Into<String>,
    min: Option<impl Into<String>>,
    max: Option<impl Into<String>>,
) -> Error {
    Error::Validation(ValidationError::OutOfRange {
        value: value.into(),
        min: min.map(Into::into),
        max: max.map(Into::into),
    })
}

/// Creates a processing internal error.
pub fn internal_error(reason: impl Into<String>) -> Error {
    Error::Processing(ProcessingError::Internal {
        reason: reason.into(),
    })
}

/// Creates an I/O error (std only).
#[cfg(feature = "std")]
pub fn io_error(path: impl Into<String>, reason: impl Into<String>) -> Error {
    Error::Io(IoError::ReadError {
        path: path.into(),
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = Error::Plugin(PluginError::NotFound { 
            name: "test-plugin".to_string() 
        });
        assert_eq!(err.to_string(), "Plugin error: Plugin 'test-plugin' not found");
    }
    
    #[test]
    fn test_error_context() {
        let err = plugin_not_found("test")
            .context(ErrorContext::new("loading plugin")
                .with_context("during startup"));
        
        let display = format!("{}", err);
        assert!(display.contains("loading plugin"));
        assert!(display.contains("during startup"));
    }
    
    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            plugin_not_found("plugin1"),
            model_not_initialized(),
        ];
        let err = Error::Multiple(errors);
        
        let display = err.to_string();
        assert!(display.contains("Multiple errors"));
        assert!(display.contains("plugin1"));
        assert!(display.contains("Model not initialized"));
    }
}