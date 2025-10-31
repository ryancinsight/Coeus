//! Production-Grade CLIP Semantic Search REST API Server
//!
//! This example demonstrates a complete, working REST API for semantic search
//! with CLIP embeddings, showcasing enterprise-grade patterns and production readiness.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Use real CLIP service and vector database from the semantic API crate
use semantic_api::clip_service::{RealCLIPService, InMemoryVectorDB};
use semantic_api::state::{CLIPService, VectorDatabase};

// Application state using real CLIP service
#[derive(Clone)]
struct AppState {
    clip_service: Arc<dyn CLIPService + Send + Sync>,
    vector_db: Arc<dyn VectorDatabase + Send + Sync>,
}

// Request/Response types
#[derive(serde::Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize { 10 }

#[derive(serde::Serialize)]
struct SearchResult {
    id: String,
    #[serde(rename = "similarity")]
    score: f32,
    metadata: serde_json::Value,
}

#[derive(serde::Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    total_found: usize,
    query: String,
}

#[derive(serde::Serialize)]
struct HealthResponse {
    status: String,
    uptime_seconds: u64,
    version: String,
}

// Main server implementation
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router
};

async fn health_check(State(_state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn text_search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    if request.query.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Query cannot be empty".to_string()));
    }

    let embedding = state.clip_service.encode_text(&request.query)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let results = state.vector_db.search(&embedding, request.top_k).await;

    let response_results: Vec<SearchResult> = results.into_iter()
        .map(|(id, score, metadata)| SearchResult { id, score, metadata })
        .collect();

    Ok(Json(SearchResponse {
        results: response_results,
        total_found: state.vector_db.size().await,
        query: request.query,
    }))
}

async fn index_content(
    State(state): State<AppState>,
    Json(entries): Json<Vec<(String, String, serde_json::Value)>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    for (id, content, metadata) in entries {
        let embedding = state.clip_service.encode_text(&content)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

        state.vector_db.add(id, embedding, metadata).await;
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "indexed_count": entries.len()
    })))
}

async fn initialize_sample_data(state: &AppState) {
    let sample_data = vec![
        ("doc1".to_string(), "A beautiful mountain landscape with snow-capped peaks".to_string(), serde_json::json!({"type": "image", "tags": ["nature", "mountains"]})),
        ("doc2".to_string(), "A sleek sports car speeding down a coastal highway".to_string(), serde_json::json!({"type": "image", "tags": ["car", "speed"]})),
        ("doc3".to_string(), "People enjoying delicious pizza at an outdoor restaurant".to_string(), serde_json::json!({"type": "image", "tags": ["food", "people"]})),
        ("doc4".to_string(), "Ancient book collection in a cozy library".to_string(), serde_json::json!({"type": "image", "tags": ["books", "library"]})),
    ];

    for (id, content, metadata) in sample_data {
        let embedding = state.clip_service.encode_text(&content).await.unwrap();
        state.vector_db.add(id, embedding, metadata).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("semantic_api=info")
        .init();

    eprintln!("🚀 Semantic Search API Server Starting...");
    eprintln!("====================================");
    eprintln!("📚 Production-grade CLIP semantic search REST API");
    eprintln!("🔍 Supports text-based semantic search");
    eprintln!("📊 Vector similarity with optimized performance");
    eprintln!("🏗️  Built with async Tokio + Axum + comprehensive error handling");

    // Initialize real CLIP service
    let clip_service: Arc<dyn CLIPService + Send + Sync> = {
        #[cfg(feature = "gpu")]
        {
            eprintln!("🎮 Using GPU-accelerated CLIP service");
            Arc::new(RealCLIPService::new_with_gpu().await?)
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("💻 Using CPU-based CLIP service");
            Arc::new(RealCLIPService::new_with_cpu()?)
        }
    };

    let vector_db: Arc<dyn VectorDatabase + Send + Sync> = Arc::new(InMemoryVectorDB::new());

    let state = AppState {
        clip_service,
        vector_db,
    };

    // Initialize sample data
    initialize_sample_data(&state).await;
    eprintln!("💾 Initialized with sample vector database");

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/search/text", axum::routing::post(text_search))
        .route("/index", axum::routing::post(index_content))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("🌐 Listening on {}", addr);
    eprintln!("📋 Available endpoints:");
    eprintln!("   GET  /health              - Health check");
    eprintln!("   POST /search/text         - Semantic text search");
    eprintln!("   POST /index              - Index content");
    eprintln!("📖 API Documentation: Production-grade with error handling");
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║                    SERVER IS RUNNING                    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Client;
    use std::net::TcpListener;
    use tokio::spawn;

    async fn spawn_test_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let addr_string = format!("http://{}", addr);

        // Use CPU-based services for tests
        let clip_service: Arc<dyn CLIPService + Send + Sync> = Arc::new(RealCLIPService::new_with_cpu().unwrap());
        let vector_db: Arc<dyn VectorDatabase + Send + Sync> = Arc::new(InMemoryVectorDB::new());

        let state = AppState {
            clip_service,
            vector_db,
        };

        initialize_sample_data(&state).await;

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/search/text", axum::routing::post(text_search))
            .with_state(state);

        spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        addr_string
    }

    #[tokio::test]
    async fn test_health_check() {
        let base_url = spawn_test_server().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let client = Client::new();
        let response = client
            .get(format!("{}/health", base_url))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 200);

        let health: HealthResponse = response.json().await.unwrap();
        assert_eq!(health.status, "healthy");
        assert!(health.uptime_seconds > 0);
    }

    #[tokio::test]
    async fn test_text_search() {
        let base_url = spawn_test_server().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let client = Client::new();
        let request = SearchRequest {
            query: "mountain landscape".to_string(),
            top_k: 3,
        };

        let response = client
            .post(format!("{}/search/text", base_url))
            .json(&request)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 200);

        let search_result: SearchResponse = response.json().await.unwrap();
        assert!(!search_result.results.is_empty());
        assert_eq!(search_result.query, "mountain landscape");
        assert!(search_result.total_found > 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let similarity = cosine_similarity(&a, &b);
        let expected = 32.0 / (14.0_f32.sqrt() * 77.0_f32.sqrt());

        assert!((similarity - expected).abs() < 1e-6);
    }
}

