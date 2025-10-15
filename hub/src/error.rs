//! Error types for the model hub

use thiserror::Error;

/// Result type alias for hub operations
pub type Result<T> = std::result::Result<T, HubError>;

/// Errors that can occur during model hub operations
#[derive(Error, Debug)]
pub enum HubError {
    #[error("Model not found: {name}")]
    ModelNotFound { name: String },

    #[error("Model version not found: {name}@{version}")]
    VersionNotFound { name: String, version: String },

    #[error("Model download failed: {url} - {message}")]
    DownloadFailed { url: String, message: String },

    #[error("Model validation failed: {model} - {reason}")]
    ValidationFailed { model: String, reason: String },

    #[error("Cache operation failed: {message}")]
    CacheError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Model incompatible with requested task: {model} cannot perform {task}")]
    TaskMismatch { model: String, task: String },

    #[error("Model file corrupted: {model} - checksum mismatch")]
    CorruptedModel { model: String },

    #[error("Model loading failed: {model} - {reason}")]
    LoadingFailed { model: String, reason: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Registry error: {message}")]
    RegistryError { message: String },

    #[error("HTTP error: {status} - {message}")]
    HttpError { status: u16, message: String },

    #[error("JSON parsing error: {message}")]
    JsonError { message: String },

    #[error("Invalid model metadata: {field} - {reason}")]
    InvalidMetadata { field: String, reason: String },

    #[error("Model size exceeds cache limit: {requested} > {limit}")]
    CacheLimitExceeded { requested: u64, limit: u64 },
}
