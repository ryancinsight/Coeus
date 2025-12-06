//! Coeus Semantic Search REST API
//!
//! Production-grade REST API for CLIP semantic search with comprehensive
//! error handling, monitoring, and enterprise features.

use std::sync::Arc;

use axum::{
    extract::{self, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
    middleware,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, request_id::MakeRequestUuid,
    timeout::TimeoutLayer, trace::TraceLayer,
};

// Re-export for convenience
pub use crate::types::*;
pub use crate::state::*;
pub use crate::handlers::*;
pub use crate::custom_middleware::*;
pub use crate::errors::*;

// Add async_trait for async trait methods
#[macro_use]
extern crate async_trait;

/// Core API module
pub mod types;
/// Application state management
pub mod state;
/// CLIP service implementation
pub mod clip_service;
/// Request handlers
pub mod handlers;
/// Custom middleware
pub mod custom_middleware;
/// Error handling
pub mod errors;

/// Create the complete REST API router with all endpoints and middleware
pub fn create_router() -> Router<AppState> {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/search/text", post(text_search))
        .route("/v1/search/image", post(image_search))
        .route("/v1/search/cross-modal", post(cross_modal_search))
        .route("/v1/index", post(index_content))
        .route("/v1/benchmarks", get(benchmark_search))
        .route("/v1/metrics", get(get_metrics))
        .layer(middleware::from_fn(|req, next| async {
            custom_middleware::metrics_middleware(req, next).await
        }))
        .layer(middleware::from_fn(|req, next| async {
            custom_middleware::request_id_middleware(req, next).await
        }))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(
                    CorsLayer::new()
                        .allow_origin(tower_http::cors::Any)
                        .allow_methods(tower_http::cors::Any)
                        .allow_headers(tower_http::cors::Any),
                )
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(tower_http::request_id::PropagateRequestIdLayer::x_request_id())
                .layer(tower_http::request_id::SetRequestIdLayer::x_request_id(
                    MakeRequestUuid::default(),
                )),
        )
}

/// Initialize and start the semantic search API server
pub async fn start_server(state: AppState, host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;
    let router = create_router().with_state(state);

    tracing::info!("🚀 Semantic Search API starting on {}", addr);
    axum::serve(listener, router).await?;

    Ok(())
}

/// Initialize tracing for observability
pub fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::{filter::LevelFilter, EnvFilter};

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env()?; // Allow override via RUST_LOG

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .with_file(true)
        .with_line_number(true)
        .init();

    Ok(())
}

/// Initialize metrics for monitoring
pub fn init_metrics() -> Result<(), Box<dyn std::error::Error>> {
    use metrics_exporter_prometheus::Matcher;

    // Initialize global recorder for Prometheus metrics
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let _recorder = builder.install_recorder()?;
    // metrics::set_global_recorder is not needed as install_recorder already sets it

    // Register standard metrics
    metrics::counter!("requests_total");
    metrics::histogram!("request_duration_seconds");
    metrics::gauge("search_index_size").set(0.0);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use axum::body::Body;
    use axum::extract::Request;
    use http::StatusCode;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let state = AppState::new_for_testing().unwrap();
        let router = create_router().with_state(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_router_creation() {
        let state = AppState::new_for_testing().unwrap();
        let router = create_router().with_state(state);

        // Router should be created without panicking
        assert!(router.ready().await.is_ok());
    }
}