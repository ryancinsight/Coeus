//! Error types for the hub crate

/// Errors that can occur during hub operations
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// Network-related errors
    #[error("Network error: {source}")]
    Network {
        #[from]
        source: reqwest::Error,
    },

    /// I/O errors
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// JSON parsing errors
    #[error("JSON parsing error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    /// Pickle parsing errors
    #[error("Pickle parsing error: {source}")]
    Pickle {
        #[from]
        source: serde_pickle::Error,
    },

    /// URL parsing errors
    #[error("URL parsing error: {source}")]
    Url {
        #[from]
        source: url::ParseError,
    },

    /// Download errors
    #[error("Download failed for {url}: {message}")]
    DownloadError { url: String, message: String },

    /// Invalid file format
    #[error("Invalid file format for {filename}: {message}")]
    InvalidFileFormat { filename: String, message: String },

    /// Model not found
    #[error("Model '{repo}/{model}' not found")]
    ModelNotFound { repo: String, model: String },

    /// Invalid model format
    #[error("Invalid model format: {message}")]
    InvalidModelFormat { message: String },

    /// State dict loading error
    #[error("State dict error: {message}")]
    StateDictError { message: String },

    /// Hash verification failed
    #[error("Hash verification failed: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    /// Cache directory creation failed
    #[error("Cache directory error: {message}")]
    CacheError { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Untrusted repository
    #[error("Repository '{repo}' is not trusted")]
    UntrustedRepository { repo: String },

    /// Generic error
    #[error("Hub error: {message}")]
    Other { message: String },
}

impl HubError {
    /// Create a new other error
    pub fn other<S: Into<String>>(message: S) -> Self {
        Self::Other {
            message: message.into(),
        }
    }

    /// Create a new state dict error
    pub fn state_dict<S: Into<String>>(message: S) -> Self {
        Self::StateDictError {
            message: message.into(),
        }
    }

    /// Create a new invalid model format error
    pub fn invalid_format<S: Into<String>>(message: S) -> Self {
        Self::InvalidModelFormat {
            message: message.into(),
        }
    }

    /// Create a new download error
    pub fn download_error<S: Into<String>>(url: String, message: S) -> Self {
        Self::DownloadError {
            url,
            message: message.into(),
        }
    }

    /// Create a new invalid file format error
    pub fn invalid_file_format<S: Into<String>>(filename: String, message: S) -> Self {
        Self::InvalidFileFormat {
            filename,
            message: message.into(),
        }
    }
}
