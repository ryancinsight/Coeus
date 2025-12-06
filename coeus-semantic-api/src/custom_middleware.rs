//! Custom Middleware for Semantic Search API
//!
//! Enterprise-grade middleware for request processing, metrics collection,
//! rate limiting, authentication, and observability.

use axum::{
    extract::Request,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    middleware::Next,
};
// use tower::{Layer, Service};
// use std::task::{Context, Poll};
use std::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Request ID middleware - assigns unique ID to each request
pub async fn request_id_middleware(request: Request, next: Next) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();

    // Add request ID to response headers
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        header::HeaderValue::from_str(&request_id).unwrap(),
    );

    response
}

/// Metrics collection middleware
pub async fn metrics_middleware(request: Request, next: Next) -> Response {
    let start_time = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;

    let duration = start_time.elapsed();
    let status_code = response.status();

    // Record metrics
    metrics::counter!("requests_total", "method" => method.to_string(), "status" => status_code.to_string()).increment(1);
    metrics::histogram!("request_duration_seconds", "method" => method.to_string(), "endpoint" => uri.path().to_string())
        .record(duration.as_secs_f64());

    response
}

/// Rate limiting middleware
pub fn rate_limit_middleware(
    state: Arc<RwLock<HashMap<String, RateLimitInfo>>>,
    requests_per_minute: u32,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> + Clone + Send {
    move |request: Request, next: Next| {
        let state = state.clone();
        Box::pin(async move {
            let client_ip = extract_client_ip(&request);
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let mut rate_limits = state.write().await;

            let info = rate_limits.entry(client_ip.clone()).or_insert(RateLimitInfo {
                request_count: 0,
                window_start: current_time,
            });

            // Reset counter if window has passed
            if current_time - info.window_start >= 60 {
                info.request_count = 0;
                info.window_start = current_time;
            }

            // Check rate limit
            if info.request_count >= requests_per_minute {
                let retry_after = 60 - (current_time - info.window_start);
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::RETRY_AFTER, retry_after.to_string())],
                    "Rate limit exceeded. Please try again later.",
                ).into_response();
            }

            info.request_count += 1;
            drop(rate_limits);

            next.run(request).await
        })
    }
}

/// Authentication middleware (basic token-based auth)
pub async fn auth_middleware(
    valid_tokens: Arc<HashMap<String, String>>,
    request: Request,
    next: Next,
) -> Response {
    // Check for authorization header
    let auth_header = request.headers().get(header::AUTHORIZATION);

    if let Some(auth_value) = auth_header {
        if let Ok(auth_str) = auth_value.to_str() {
            if auth_str.starts_with("Bearer ") {
                let token = &auth_str[7..]; // Remove "Bearer " prefix

                // Check if token is valid
                if valid_tokens.contains_key(token) {
                    return next.run(request).await;
                }
            }
        }
    }

    // Authentication failed
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        "Invalid or missing authentication token",
    ).into_response()
}

/// Request logging middleware
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let version = request.version();
    let user_agent = request.headers()
        .get(header::USER_AGENT)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    tracing::info!(
        method = %method,
        uri = %uri,
        version = ?version,
        user_agent = %user_agent,
        "Request received"
    );

    let start_time = Instant::now();
    let response = next.run(request).await;
    let duration = start_time.elapsed();

    let status = response.status();

    tracing::info!(
        method = %method,
        uri = %uri,
        status = %status,
        duration_ms = duration.as_millis(),
        "Request completed"
    );

    response
}

/// CORS preflight handler
pub async fn cors_preflight_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            (header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"),
            (header::ACCESS_CONTROL_ALLOW_METHODS, "GET, POST, PUT, DELETE, OPTIONS"),
            (header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, Authorization, X-Requested-With"),
            (header::ACCESS_CONTROL_MAX_AGE, "86400"),
        ],
    )
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();
    headers.insert(header::X_CONTENT_TYPE_OPTIONS, header::HeaderValue::from_static("nosniff"));
    headers.insert(header::X_FRAME_OPTIONS, header::HeaderValue::from_static("DENY"));
    headers.insert(header::X_XSS_PROTECTION, header::HeaderValue::from_static("1; mode=block"));
    headers.insert(header::REFERRER_POLICY, header::HeaderValue::from_static("strict-origin-when-cross-origin"));
    headers.insert(header::CONTENT_SECURITY_POLICY, header::HeaderValue::from_static("default-src 'self'"));

    response
}

/// Request timeout middleware
pub fn timeout_middleware(
    timeout_duration: std::time::Duration,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> + Clone {
    move |request: Request, next: Next| {
        let timeout_duration = timeout_duration;
        Box::pin(async move {
            let timeout_future = tokio::time::timeout(timeout_duration, next.run(request));
            match timeout_future.await {
                Ok(response) => response,
                Err(_) => (
                    StatusCode::REQUEST_TIMEOUT,
                    "Request timeout exceeded",
                ).into_response(),
            }
        })
    }
}

/// Cache control middleware
pub fn cache_middleware(
    cache_enabled: bool,
    cache_ttl: u64,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> + Clone {
    move |request: Request, next: Next| {
        let cache_enabled = cache_enabled;
        let cache_ttl = cache_ttl;
        Box::pin(async move {
            let mut response = next.run(request).await;

        if cache_enabled {
            let cache_control = format!("public, max-age={}", cache_ttl);
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                header::HeaderValue::from_str(&cache_control).unwrap(),
            );
        } else {
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                header::HeaderValue::from_static("no-cache, no-store, must-revalidate"),
            );
        }

        response
        })
    }
}

/// Request size limiting middleware
pub async fn request_size_limit_middleware(
    max_size_bytes: usize,
    request: Request,
    next: Next,
) -> Response {
    // For this implementation, we'll check Content-Length header
    // In production, you'd want to stream and check actual bytes
    if let Some(content_length) = request.headers().get(header::CONTENT_LENGTH) {
        if let Ok(size) = content_length.to_str().unwrap_or("0").parse::<usize>() {
            if size > max_size_bytes {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("Request size {} bytes exceeds limit of {} bytes", size, max_size_bytes),
                ).into_response();
            }
        }
    }

    next.run(request).await
}

/// Extract client IP address from request
fn extract_client_ip(request: &Request) -> String {
    // Check X-Forwarded-For header (for proxies/load balancers)
    if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded_for.to_str() {
            // Take first IP in case of multiple
            if let Some(first_ip) = forwarded_str.split(',').next() {
                return first_ip.trim().to_string();
            }
        }
    }

    // Check X-Real-IP header
    if let Some(real_ip) = request.headers().get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }

    // Fallback to connection info (would need tower Http extension)
    // For now, return a default
    "unknown".to_string()
}

/// Rate limit tracking information
#[derive(Debug, Clone)]
struct RateLimitInfo {
    request_count: u32,
    window_start: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::Request;
    use http::Method;

    #[tokio::test]
    async fn test_request_id_middleware() {
        let request = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let next = Next::new(|_| async { "test response".into_response() });
        let response = request_id_middleware(request, next).await;

        assert!(response.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn test_extract_client_ip() {
        let request = Request::builder()
            .method(Method::GET)
            .header("x-forwarded-for", "192.168.1.100, 10.0.0.1")
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let ip = extract_client_ip(&request);
        assert_eq!(ip, "192.168.1.100");
    }
}





