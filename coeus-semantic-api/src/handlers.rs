//! Request Handlers for Semantic Search API
//!
//! Complete implementation of all REST API endpoints with proper error handling,
//! metrics collection, and enterprise-grade response formatting.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::types::*;
use crate::state::*;

/// Health check endpoint
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let start_time = Instant::now();

    // Update health status
    if let Err(_) = state.update_health().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::ServiceUnavailable,
                message: "Health check failed".to_string(),
                details: None,
                retry_after: Some(30),
            }),
        );
    }

    let health = state.health.read().await;
    let processing_time = start_time.elapsed().as_millis() as u64;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            "health-check".to_string(),
            health.clone(),
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Text-based semantic search
pub async fn text_search(
    State(state): State<AppState>,
    Json(request): Json<TextSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // Validate request
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: "Query cannot be empty".to_string(),
                details: None,
                retry_after: None,
            }),
        );
    }

    let top_k = request.top_k.unwrap_or(state.config.default_top_k);
    if top_k > state.config.max_top_k {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: format!("top_k cannot exceed {}", state.config.max_top_k),
                details: None,
                retry_after: None,
            }),
        );
    }

    // Generate text embedding
    let embedding = match state.clip_service.encode_text(&request.query).await {
        Ok(emb) => emb,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Failed to encode text: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    // Perform similarity search
    let results = match state.vector_db.search(&embedding, top_k).await {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Search failed: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    let processing_time = start_time.elapsed().as_millis() as u64;

    // Record metrics
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            TextSearchResponse {
                query: request.query,
                results: results.into_iter().map(|r| SearchResultItem {
                    id: r.id,
                    similarity: r.similarity,
                    metadata: r.metadata,
                }).collect(),
                total_results: results.len(),
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Image-based semantic search
pub async fn image_search(
    State(state): State<AppState>,
    Json(request): Json<ImageSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let top_k = request.top_k.unwrap_or(state.config.default_top_k);
    if top_k > state.config.max_top_k {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: format!("top_k cannot exceed {}", state.config.max_top_k),
                details: None,
                retry_after: None,
            }),
        );
    }

    // Generate image embedding
    let embedding = match state.clip_service.encode_image(&request.image_data).await {
        Ok(emb) => emb,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Failed to encode image: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    // Perform similarity search
    let results = match state.vector_db.search(&embedding, top_k).await {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Search failed: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    let processing_time = start_time.elapsed().as_millis() as u64;

    // Record metrics
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            ImageSearchResponse {
                results: results.into_iter().map(|r| SearchResultItem {
                    id: r.id,
                    similarity: r.similarity,
                    metadata: r.metadata,
                }).collect(),
                total_results: results.len(),
                embedding_dim: embedding.len(),
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Cross-modal search (text query, image results or vice versa)
pub async fn cross_modal_search(
    State(state): State<AppState>,
    Json(request): Json<CrossModalSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let top_k = request.top_k.unwrap_or(state.config.default_top_k);
    if top_k > state.config.max_top_k {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: format!("top_k cannot exceed {}", state.config.max_top_k),
                details: None,
                retry_after: None,
            }),
        );
    }

    // Generate embedding based on query type
    let embedding = match &request.query {
        CrossModalQuery::Text { text } => {
            if text.trim().is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error_id: uuid::Uuid::new_v4().to_string(),
                        code: ErrorCode::InvalidRequest,
                        message: "Text query cannot be empty".to_string(),
                        details: None,
                        retry_after: None,
                    }),
                );
            }
            state.clip_service.encode_text(text).await
        }
        CrossModalQuery::Image { data } => {
            state.clip_service.encode_image(data).await
        }
    };

    let embedding = match embedding {
        Ok(emb) => emb,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Failed to encode query: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    // Perform similarity search
    let results = match state.vector_db.search(&embedding, top_k).await {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InternalError,
                    message: format!("Search failed: {}", e),
                    details: None,
                    retry_after: None,
                }),
            );
        }
    };

    let processing_time = start_time.elapsed().as_millis() as u64;

    // Record metrics
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            CrossModalSearchResponse {
                query_type: match request.query {
                    CrossModalQuery::Text { .. } => "text".to_string(),
                    CrossModalQuery::Image { .. } => "image".to_string(),
                },
                results: results.into_iter().map(|r| SearchResultItem {
                    id: r.id,
                    similarity: r.similarity,
                    metadata: r.metadata,
                }).collect(),
                total_results: results.len(),
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Index new content for search
pub async fn index_content(
    State(state): State<AppState>,
    Json(request): Json<IndexRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // Validate request
    if request.items.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: "No items to index".to_string(),
                details: None,
                retry_after: None,
            }),
        );
    }

    if request.items.len() > state.config.max_batch_size {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: format!("Batch size cannot exceed {}", state.config.max_batch_size),
                details: None,
                retry_after: None,
            }),
        );
    }

    let mut indexed_count = 0;
    let mut errors = Vec::new();

    // Process each item
    for item in &request.items {
        // Generate embedding based on content type
        let embedding = match &item.content {
            IndexContent::Text { text } => {
                if text.trim().is_empty() {
                    errors.push(format!("Empty text for item {}", item.id));
                    continue;
                }
                state.clip_service.encode_text(text).await
            }
            IndexContent::Image { data } => {
                state.clip_service.encode_image(data).await
            }
        };

        let embedding = match embedding {
            Ok(emb) => emb,
            Err(e) => {
                errors.push(format!("Failed to encode item {}: {}", item.id, e));
                continue;
            }
        };

        // Add to vector database
        if let Err(e) = state.vector_db.add(
            item.id.clone(),
            embedding,
            item.metadata.clone(),
        ).await {
            errors.push(format!("Failed to index item {}: {}", item.id, e));
            continue;
        }

        indexed_count += 1;
        state.record_index_operation().await;
    }

    let processing_time = start_time.elapsed().as_millis() as u64;
    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            IndexResponse {
                indexed_count,
                total_requested: request.items.len(),
                errors,
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Get search performance benchmarks
pub async fn benchmark_search(
    State(state): State<AppState>,
    Query(params): Query<BenchmarkQuery>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // Generate test queries
    let num_queries = params.num_queries.unwrap_or(10);
    let test_queries = generate_test_queries(num_queries);

    let mut latencies = Vec::new();
    let mut results_counts = Vec::new();

    // Run benchmark queries
    for query in test_queries {
        let query_start = Instant::now();

        let embedding = match state.clip_service.encode_text(&query).await {
            Ok(emb) => emb,
            Err(_) => continue, // Skip failed encodings
        };

        let results = match state.vector_db.search(&embedding, params.top_k.unwrap_or(10)).await {
            Ok(res) => res,
            Err(_) => continue, // Skip failed searches
        };

        let latency = query_start.elapsed().as_micros() as f64;
        latencies.push(latency);
        results_counts.push(results.len());
    }

    // Calculate statistics
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50_latency = percentile(&latencies, 50.0);
    let p95_latency = percentile(&latencies, 95.0);
    let p99_latency = percentile(&latencies, 99.0);

    let processing_time = start_time.elapsed().as_millis() as u64;
    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            BenchmarkResponse {
                num_queries: latencies.len(),
                avg_latency_us: avg_latency,
                p50_latency_us: p50_latency,
                p95_latency_us: p95_latency,
                p99_latency_us: p99_latency,
                throughput_qps: latencies.len() as f64 / (processing_time as f64 / 1000.0),
                avg_results_count: results_counts.iter().sum::<usize>() as f64 / results_counts.len() as f64,
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Get service metrics
pub async fn get_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let start_time = Instant::now();

    let metrics = state.get_metrics().await;
    let db_stats = match state.vector_db.stats().await {
        Ok(stats) => Some(stats),
        Err(_) => None,
    };

    let processing_time = start_time.elapsed().as_millis() as u64;
    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            MetricsResponse {
                search_metrics: metrics,
                database_stats: db_stats,
                uptime_seconds: health.uptime_seconds,
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
}

/// Generate test queries for benchmarking
fn generate_test_queries(count: usize) -> Vec<String> {
    let templates = vec![
        "a photo of a {}",
        "an image showing {}",
        "picture of {}",
        "beautiful {}",
        "close-up of {}",
    ];

    let objects = vec![
        "cat", "dog", "car", "house", "tree", "person", "bicycle", "bird", "flower", "mountain",
        "ocean", "sunset", "city", "building", "food", "animal", "vehicle", "landscape", "portrait", "nature",
    ];

    (0..count).map(|i| {
        let template = &templates[i % templates.len()];
        let object = &objects[i % objects.len()];
        template.replace("{}", object)
    }).collect()
}

/// Calculate percentile from a sorted vector
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = (p / 100.0 * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}





