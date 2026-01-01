//! Error Handling for Semantic Search API
//!
//! Comprehensive error types and handling for production-grade API
//! with proper error propagation, logging, and user-friendly messages.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
// use serde::Serialize;
// use std::fmt;
use thiserror::Error;

/// Custom error type for semantic search operations
#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("CLIP encoding failed: {0}")]
    ClipEncodingError(String),

    #[error("Vector database error: {0}")]
    DatabaseError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Authorization failed: {0}")]
    AuthorizationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Metrics error: {0}")]
    MetricsError(String),
}

impl SemanticError {
    /// Convert error to appropriate HTTP status code and error code
    pub fn to_status_code_and_error_code(&self) -> (StatusCode, crate::types::ErrorCode) {
        match self {
            SemanticError::InvalidInput(_) => (
                StatusCode::BAD_REQUEST,
                crate::types::ErrorCode::InvalidRequest,
            ),
            SemanticError::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                crate::types::ErrorCode::RateLimited,
            ),
            SemanticError::AuthenticationError(_) => (
                StatusCode::UNAUTHORIZED,
                crate::types::ErrorCode::Unauthorized,
            ),
            SemanticError::AuthorizationError(_) => {
                (StatusCode::FORBIDDEN, crate::types::ErrorCode::Forbidden)
            }
            SemanticError::ServiceUnavailable(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                crate::types::ErrorCode::ServiceUnavailable,
            ),
            SemanticError::ClipEncodingError(_) | SemanticError::DatabaseError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                crate::types::ErrorCode::InternalError,
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                crate::types::ErrorCode::InternalError,
            ),
        }
    }

    /// Check if this error should be logged as a warning rather than error
    pub fn is_warning(&self) -> bool {
        matches!(self, SemanticError::RateLimitExceeded)
    }

    /// Get retry recommendation for this error
    pub fn retry_after_seconds(&self) -> Option<u32> {
        match self {
            SemanticError::RateLimitExceeded => Some(60),
            SemanticError::ServiceUnavailable(_) => Some(30),
            _ => None,
        }
    }
}

/// Convert SemanticError to Axum response
impl IntoResponse for SemanticError {
    fn into_response(self) -> Response {
        let (status_code, error_code) = self.to_status_code_and_error_code();

        // Log the error appropriately
        if self.is_warning() {
            tracing::warn!("API error: {}", self);
        } else {
            tracing::error!("API error: {}", self);
        }

        // Record error metrics
        metrics::counter!("errors_total", "code" => format!("{:?}", error_code)).increment(1);

        let error_response = crate::types::ErrorResponse {
            error_id: uuid::Uuid::new_v4().to_string(),
            code: error_code,
            message: self.to_string(),
            details: None, // Could add more details in production
            retry_after: self.retry_after_seconds(),
        };

        (status_code, Json(error_response)).into_response()
    }
}

/// Result type alias for semantic operations
pub type SemanticResult<T> = Result<T, SemanticError>;

/// Error handling utilities
pub struct ErrorHandler;

impl ErrorHandler {
    fn handle_error_message(error_msg: String, context: &str) -> SemanticError {
        let error_msg = format!("{}: {}", context, error_msg);

        if error_msg.contains("rate limit") {
            tracing::warn!("Rate limit triggered: {}", error_msg);
            SemanticError::RateLimitExceeded
        } else if error_msg.contains("auth") {
            tracing::warn!("Authentication error: {}", error_msg);
            SemanticError::AuthenticationError(error_msg)
        } else {
            tracing::error!("Unexpected error in {}: {}", context, error_msg);
            SemanticError::InternalError(error_msg)
        }
    }

    /// Handle and log errors with appropriate severity
    pub fn handle_error<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        context: &str,
    ) -> SemanticError {
        Self::handle_error_message(error.to_string(), context)
    }

    /// Convert any error to SemanticError
    pub fn convert_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> SemanticError {
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(error);

        let boxed = match boxed.downcast::<std::io::Error>() {
            Ok(io_err) => return SemanticError::IoError(*io_err),
            Err(boxed) => boxed,
        };

        let boxed = match boxed.downcast::<serde_json::Error>() {
            Ok(json_err) => return SemanticError::JsonError(*json_err),
            Err(boxed) => boxed,
        };

        let boxed = match boxed.downcast::<reqwest::Error>() {
            Ok(http_err) => return SemanticError::HttpError(*http_err),
            Err(boxed) => boxed,
        };

        Self::handle_error_message(boxed.to_string(), "operation")
    }

    /// Create a service unavailable error with context
    pub fn service_unavailable(context: &str) -> SemanticError {
        let msg = format!("Service temporarily unavailable: {}", context);
        tracing::warn!("{}", msg);
        SemanticError::ServiceUnavailable(msg)
    }

    /// Create an invalid input error
    pub fn invalid_input(field: &str, reason: &str) -> SemanticError {
        let msg = format!("Invalid {}: {}", field, reason);
        tracing::debug!("Validation error: {}", msg);
        SemanticError::InvalidInput(msg)
    }

    /// Log successful operations for monitoring
    pub fn log_success(operation: &str, duration_ms: u64) {
        tracing::info!(
            "Operation '{}' completed successfully in {}ms",
            operation,
            duration_ms
        );
        metrics::histogram!("operation_duration_ms", "operation" => operation.to_string())
            .record(duration_ms as f64);
    }

    /// Log operation failures
    pub fn log_failure(operation: &str, error: &SemanticError, duration_ms: u64) {
        tracing::error!(
            "Operation '{}' failed after {}ms: {}",
            operation,
            duration_ms,
            error
        );
        metrics::counter!("operation_failures_total", "operation" => operation.to_string(), "error_type" => format!("{:?}", error)).increment(1);
    }
}

/// Panic handler for graceful shutdown
pub fn set_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}", loc.file(), loc.line()))
            .unwrap_or_else(|| "unknown location".to_string());

        let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };

        tracing::error!(
            location = %location,
            payload = %payload,
            "Panic occurred, shutting down gracefully"
        );

        // In production, you might want to:
        // - Send alerts to monitoring systems
        // - Attempt graceful shutdown of services
        // - Write panic info to crash logs

        metrics::counter!("panics_total").increment(1);
    }));
}

/// Graceful shutdown handler
pub async fn shutdown_signal() -> std::io::Result<()> {
    use tokio::signal;

    let ctrl_c = async { signal::ctrl_c().await };

    #[cfg(unix)]
    let terminate = async {
        let mut sig = signal::unix::signal(signal::unix::SignalKind::terminate())?;
        sig.recv().await;
        Ok(())
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<std::io::Result<()>>();

    tokio::select! {
        res = ctrl_c => { res?; }
        res = terminate => { res?; }
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let semantic_error = ErrorHandler::convert_error(io_error);

        match semantic_error {
            SemanticError::IoError(_) => {} // Expected
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_error_status_codes() {
        let invalid_input = SemanticError::InvalidInput("test".to_string());
        let (status, code) = invalid_input.to_status_code_and_error_code();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(matches!(code, crate::types::ErrorCode::InvalidRequest));

        let rate_limit = SemanticError::RateLimitExceeded;
        let (status, code) = rate_limit.to_status_code_and_error_code();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(matches!(code, crate::types::ErrorCode::RateLimited));
    }

    #[test]
    fn test_error_retry_logic() {
        let rate_limit = SemanticError::RateLimitExceeded;
        assert_eq!(rate_limit.retry_after_seconds(), Some(60));

        let invalid_input = SemanticError::InvalidInput("test".to_string());
        assert_eq!(invalid_input.retry_after_seconds(), None);
    }

    #[tokio::test]
    async fn test_error_response_conversion() {
        let error = SemanticError::InvalidInput("test input".to_string());
        let response = error.into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
