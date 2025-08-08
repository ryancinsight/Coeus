//! Error handling for `RustLLM` Core.
//!
//! This module provides a zero-dependency error handling system that follows
//! Rust best practices while maintaining simplicity and performance.
//!
//! ## Design Principles
//!
//! - **Atomicity**: Errors are complete and self-contained
//! - **Consistency**: Error types follow a consistent pattern
//! - **Isolation**: Errors don't leak implementation details
//! - **Durability**: Error information is preserved through the stack
//! - **Zero-cost**: No runtime overhead for the happy path

use core::fmt;

/// The main result type for `RustLLM` Core.
pub type Result<T> = core::result::Result<T, Error>;

/// The main error type for `RustLLM` Core.
///
/// This type follows the principle of exhaustive error handling,
/// where each variant represents a specific error category.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// Plugin system errors.
    Plugin(PluginError),

    /// Data processing errors (tokenizer, model, etc.).
    Processing(ProcessingError),

    /// I/O errors (when std feature is enabled).
    #[cfg(feature = "std")]
    Io(IoError),

    /// Configuration and validation errors.
    Config(ConfigError),

    /// Resource management errors.
    Resource(ResourceError),

    /// Composite error for multiple failures.
    Composite(CompositeError),
}

/// Error context providing rich debugging information.
///
/// This follows the principle of preserving error context
/// for better debugging and error recovery.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation that was being performed.
    pub operation: &'static str,

    /// Source location (file, line, column).
    pub location: Option<Location>,

    /// Additional context information.
    pub details: Option<String>,

    /// Timestamp when the error occurred (if available).
    #[cfg(feature = "std")]
    pub timestamp: Option<std::time::SystemTime>,
}

/// Source location information.
#[derive(Debug, Clone, Copy)]
pub struct Location {
    /// Source file.
    pub file: &'static str,
    /// Line number.
    pub line: u32,
    /// Column number.
    pub column: u32,
}

impl ErrorContext {
    /// Creates a new error context.
    pub const fn new(operation: &'static str) -> Self {
        Self {
            operation,
            location: None,
            details: None,
            #[cfg(feature = "std")]
            timestamp: None,
        }
    }

    /// Adds source location.
    #[must_use]
    pub const fn at(mut self, file: &'static str, line: u32, column: u32) -> Self {
        self.location = Some(Location { file, line, column });
        self
    }

    /// Adds additional details.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Adds timestamp (std only).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_timestamp(mut self) -> Self {
        self.timestamp = Some(std::time::SystemTime::now());
        self
    }
}

/// Macro for creating error context with location.
#[macro_export]
macro_rules! error_context {
    ($operation:expr) => {
        $crate::foundation::error::ErrorContext::new($operation).at(file!(), line!(), column!())
    };
    ($operation:expr, $details:expr) => {
        $crate::foundation::error::ErrorContext::new($operation)
            .at(file!(), line!(), column!())
            .with_details($details)
    };
}

/// Plugin system errors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PluginError {
    /// Plugin not found.
    NotFound {
        /// Plugin identifier.
        name: String,
        /// Available plugins (for suggestions).
        available: Vec<String>,
    },

    /// Plugin lifecycle error.
    Lifecycle {
        /// Plugin identifier.
        name: String,
        /// Current state.
        current_state: String,
        /// Operation that was attempted.
        operation: &'static str,
    },

    /// Plugin dependency error.
    Dependency {
        /// Plugin identifier.
        plugin: String,
        /// Missing or incompatible dependency.
        dependency: String,
        /// Version constraint if applicable.
        version_constraint: Option<String>,
    },
}

/// Processing errors (consolidated from TokenizerError and ModelError).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ProcessingError {
    /// Component not initialized.
    NotInitialized {
        /// Component type.
        component: &'static str,
    },

    /// Invalid input data.
    InvalidInput {
        /// Description of what's invalid.
        description: String,
        /// Position in input if applicable.
        position: Option<usize>,
    },

    /// Processing limit exceeded.
    LimitExceeded {
        /// Type of limit.
        limit_type: &'static str,
        /// The limit value.
        limit: usize,
        /// Actual value that exceeded the limit.
        actual: usize,
    },

    /// Shape or dimension mismatch.
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },

    /// Operation not supported.
    Unsupported {
        /// Operation description.
        operation: String,
        /// Reason if available.
        reason: Option<String>,
    },

    /// Processing failed.
    Failed {
        /// Component that failed.
        component: &'static str,
        /// Failure reason.
        reason: String,
    },
}

/// I/O errors for std environments.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum IoError {
    /// File system error.
    FileSystem {
        /// Path involved.
        path: String,
        /// Operation that failed.
        operation: IoOperation,
        /// System error if available.
        source: Option<String>,
    },

    /// Network error.
    Network {
        /// Endpoint if applicable.
        endpoint: Option<String>,
        /// Error description.
        description: String,
    },
}

/// I/O operations.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
pub enum IoOperation {
    /// Read operation.
    Read,
    /// Write operation.
    Write,
    /// Create operation.
    Create,
    /// Delete operation.
    Delete,
    /// Metadata operation.
    Metadata,
}

/// Configuration errors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ConfigError {
    /// Missing required configuration.
    Missing {
        /// Configuration key or field.
        key: String,
        /// Expected type or format.
        expected: String,
    },

    /// Invalid configuration value.
    Invalid {
        /// Configuration key or field.
        key: String,
        /// Provided value (as string).
        value: String,
        /// Validation error.
        error: String,
    },

    /// Configuration conflict.
    Conflict {
        /// First conflicting key.
        key1: String,
        /// Second conflicting key.
        key2: String,
        /// Conflict description.
        description: String,
    },
}

/// Resource management errors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ResourceError {
    /// Resource not available.
    NotAvailable {
        /// Resource type.
        resource_type: &'static str,
        /// Resource identifier.
        identifier: String,
    },

    /// Resource exhausted.
    Exhausted {
        /// Resource type.
        resource_type: &'static str,
        /// Available amount.
        available: usize,
        /// Required amount.
        required: usize,
    },

    /// Resource busy or locked.
    Busy {
        /// Resource type.
        resource_type: &'static str,
        /// Resource identifier.
        identifier: String,
        /// Holder information if available.
        holder: Option<String>,
    },
}

/// Composite error for multiple failures.
#[derive(Debug, Clone)]
pub struct CompositeError {
    /// Primary error.
    pub primary: Box<Error>,
    /// Related errors.
    pub related: Vec<Error>,
    /// Transaction ID if applicable.
    pub transaction_id: Option<String>,
}

impl CompositeError {
    /// Creates a new composite error.
    pub fn new(primary: Error) -> Self {
        Self {
            primary: Box::new(primary),
            related: Vec::new(),
            transaction_id: None,
        }
    }

    /// Adds a related error.
    pub fn with_related(mut self, error: Error) -> Self {
        self.related.push(error);
        self
    }

    /// Sets transaction ID for atomicity tracking.
    pub fn with_transaction_id(mut self, id: impl Into<String>) -> Self {
        self.transaction_id = Some(id.into());
        self
    }
}

/// Trait for adding context to errors.
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
        self.context(ErrorContext::new("operation").with_details(context))
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

// ============================================================================
// Display implementations
// ============================================================================

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plugin(e) => write!(f, "Plugin error: {e}"),
            Self::Processing(e) => write!(f, "Processing error: {e}"),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Config(e) => write!(f, "Configuration error: {e}"),
            Self::Resource(e) => write!(f, "Resource error: {e}"),
            Self::Composite(e) => {
                write!(f, "Composite error: {}", e.primary)?;
                if !e.related.is_empty() {
                    write!(f, " ({} related errors)", e.related.len())?;
                }
                Ok(())
            },
        }
    }
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { name, available } => {
                write!(f, "Plugin '{name}' not found")?;
                if !available.is_empty() {
                    write!(f, ". Available: {}", available.join(", "))?;
                }
                Ok(())
            },
            Self::Lifecycle {
                name,
                current_state,
                operation,
            } => {
                write!(
                    f,
                    "Plugin '{name}' cannot {operation} in state {current_state:?}"
                )
            },
            Self::Dependency {
                plugin,
                dependency,
                version_constraint,
            } => {
                write!(f, "Plugin '{plugin}' missing dependency '{dependency}'")?;
                if let Some(constraint) = version_constraint {
                    write!(f, " (version {constraint})")?;
                }
                Ok(())
            },
        }
    }
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized { component } => {
                write!(f, "{component} not initialized")
            },
            Self::InvalidInput {
                description,
                position,
            } => {
                write!(f, "Invalid input: {description}")?;
                if let Some(pos) = position {
                    write!(f, " at position {pos}")?;
                }
                Ok(())
            },
            Self::LimitExceeded {
                limit_type,
                limit,
                actual,
            } => {
                write!(f, "{limit_type} limit exceeded: {actual} > {limit}")
            },
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected}, got {actual}")
            },
            Self::Unsupported { operation, reason } => {
                write!(f, "Unsupported operation: {operation}")?;
                if let Some(r) = reason {
                    write!(f, " ({r})")?;
                }
                Ok(())
            },
            Self::Failed { component, reason } => {
                write!(f, "{component} failed: {reason}")
            },
        }
    }
}

#[cfg(feature = "std")]
impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileSystem {
                path,
                operation,
                source,
            } => {
                write!(f, "File system error on '{path}' during {operation:?}")?;
                if let Some(s) = source {
                    write!(f, ": {s}")?;
                }
                Ok(())
            },
            Self::Network {
                endpoint,
                description,
            } => {
                write!(f, "Network error")?;
                if let Some(ep) = endpoint {
                    write!(f, " on '{ep}'")?;
                }
                write!(f, ": {description}")
            },
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { key, expected } => {
                write!(f, "Missing configuration '{key}' (expected {expected})")
            },
            Self::Invalid { key, value, error } => {
                write!(f, "Invalid configuration '{key}' = '{value}': {error}")
            },
            Self::Conflict {
                key1,
                key2,
                description,
            } => {
                write!(
                    f,
                    "Configuration conflict between '{key1}' and '{key2}': {description}"
                )
            },
        }
    }
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAvailable {
                resource_type,
                identifier,
            } => {
                write!(f, "{resource_type} '{identifier}' not available")
            },
            Self::Exhausted {
                resource_type,
                available,
                required,
            } => {
                write!(
                    f,
                    "{resource_type} exhausted: {available} available, {required} required"
                )
            },
            Self::Busy {
                resource_type,
                identifier,
                holder,
            } => {
                write!(f, "{resource_type} '{identifier}' is busy")?;
                if let Some(h) = holder {
                    write!(f, " (held by {h})")?;
                }
                Ok(())
            },
        }
    }
}

impl<E: fmt::Display> fmt::Display for ContextualError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.context.operation, self.error)?;

        if let Some(details) = &self.context.details {
            write!(f, ": {details}")?;
        }

        if let Some(loc) = &self.context.location {
            write!(f, " at {}:{}:{}", loc.file, loc.line, loc.column)?;
        }

        #[cfg(feature = "std")]
        if let Some(ts) = &self.context.timestamp {
            if let Ok(duration) = ts.duration_since(std::time::UNIX_EPOCH) {
                write!(f, " ({})", duration.as_secs())?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Error trait implementations
// ============================================================================

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PluginError {}

#[cfg(feature = "std")]
impl std::error::Error for ProcessingError {}

#[cfg(feature = "std")]
impl std::error::Error for IoError {}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

#[cfg(feature = "std")]
impl std::error::Error for ResourceError {}

#[cfg(feature = "std")]
impl<E: std::error::Error + 'static> std::error::Error for ContextualError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Creates a not initialized error for a component.
pub fn not_initialized(component: &'static str) -> Error {
    Error::Processing(ProcessingError::NotInitialized { component })
}

/// Creates an invalid input error.
pub fn invalid_input(description: impl Into<String>) -> Error {
    Error::Processing(ProcessingError::InvalidInput {
        description: description.into(),
        position: None,
    })
}

/// Creates a limit exceeded error.
pub fn limit_exceeded(limit_type: &'static str, limit: usize, actual: usize) -> Error {
    Error::Processing(ProcessingError::LimitExceeded {
        limit_type,
        limit,
        actual,
    })
}

/// Creates a processing failed error (replacement for internal_error).
pub fn internal_error(reason: impl Into<String>) -> Error {
    Error::Processing(ProcessingError::Failed {
        component: "internal",
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("test_operation")
            .at("test.rs", 42, 10)
            .with_details("test details");

        assert_eq!(ctx.operation, "test_operation");
        assert!(ctx.location.is_some());
        assert_eq!(ctx.details.as_deref(), Some("test details"));
    }

    #[test]
    fn test_error_display() {
        let error = Error::Processing(ProcessingError::NotInitialized {
            component: "tokenizer",
        });

        let display = format!("{error}");
        assert!(display.contains("tokenizer not initialized"));
    }

    #[test]
    fn test_composite_error() {
        let primary = invalid_input("bad data");
        let related = limit_exceeded("token", 100, 150);

        let composite = CompositeError::new(primary)
            .with_related(related)
            .with_transaction_id("tx-123");

        assert_eq!(composite.related.len(), 1);
        assert_eq!(composite.transaction_id.as_deref(), Some("tx-123"));
    }

    #[test]
    fn test_error_context_macro() {
        let ctx = error_context!("macro_test");
        assert_eq!(ctx.operation, "macro_test");
        assert!(ctx.location.is_some());

        let ctx_with_details = error_context!("macro_test", "with details");
        assert_eq!(ctx_with_details.details.as_deref(), Some("with details"));
    }
}
