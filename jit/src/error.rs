//! Error types for JIT compilation operations

use thiserror::Error;

/// Result type alias for JIT operations
pub type Result<T> = std::result::Result<T, JitError>;

/// Errors that can occur during JIT compilation and graph optimization
#[derive(Error, Debug)]
pub enum JitError {
    #[error("Graph construction failed: {message}")]
    GraphConstruction { message: String },

    #[error("Optimization pass failed: {pass_name} - {message}")]
    OptimizationFailed { pass_name: String, message: String },

    #[error("Fusion detection failed: {message}")]
    FusionFailed { message: String },

    #[error("JIT compilation failed: {message}")]
    CompilationFailed { message: String },

    #[error("Invalid graph structure: {message}")]
    InvalidGraph { message: String },

    #[error("Operation not supported: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Memory allocation failed for kernel: required {required}, available {available}")]
    MemoryAllocation { required: usize, available: usize },

    #[error("Cache operation failed: {message}")]
    CacheError { message: String },

    #[error("Serialization failed: {message}")]
    SerializationError { message: String },

    #[error("Type mismatch in operation: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Kernel execution failed: {message}")]
    ExecutionFailed { message: String },

    #[error("Tracing operation failed: {message}")]
    TracingError { message: String },

    #[error("Cranelift module error: {0}")]
    ModuleError(#[from] Box<cranelift_module::ModuleError>),
}

impl From<cranelift_module::ModuleError> for JitError {
    fn from(value: cranelift_module::ModuleError) -> Self {
        Self::ModuleError(Box::new(value))
    }
}
