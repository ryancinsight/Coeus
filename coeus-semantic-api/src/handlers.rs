//! Request Handlers for Semantic Search API
//!
//! Complete implementation of all REST API endpoints with proper error handling,
//! metrics collection, and enterprise-grade response formatting.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
// use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
use base64::{engine::general_purpose, Engine as _};
use std::time::Instant;

use crate::state::AppState;
use crate::types::{
    BenchmarkQuery, ContentType, IndexContent, IndexRequest, IndexResponse, SearchConfig,
    SearchMethod, SearchMode, SearchPerformance, SearchResponse, SearchResult, *,
};
// use crate::errors::SemanticError;

/// Health check endpoint
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let start_time = Instant::now();

    // Update health status (in a real app, this might trigger active checks)
    // For now, we rely on the background health monitor

    let health = state.health.read().await;

    // Map internal ServiceStatus to API HealthStatus
    let api_status = match health.status {
        ServiceStatus::Operational => HealthStatus::Healthy,
        ServiceStatus::Degraded => HealthStatus::Degraded,
        ServiceStatus::Maintenance => HealthStatus::Unhealthy,
    };

    // Construct detailed service health if components are available
    let services = if !health.components.is_empty() {
        // Helper to get component health or default
        let get_comp = |name: &str| -> ComponentHealth {
            if let Some(status) = health.components.get(name) {
                ComponentHealth {
                    status: match status.status {
                        crate::state::ComponentStatus::Healthy => ComponentStatus::Operational,
                        crate::state::ComponentStatus::Degraded => ComponentStatus::Impaired,
                        crate::state::ComponentStatus::Unhealthy => ComponentStatus::Unavailable,
                    },
                    message: status.message.clone(),
                    version: Some(env!("CARGO_PKG_VERSION").to_string()),
                    last_check: health.last_check,
                }
            } else {
                ComponentHealth {
                    status: ComponentStatus::Unavailable,
                    message: "Component not found".to_string(),
                    version: None,
                    last_check: health.last_check,
                }
            }
        };

        Some(ServiceHealth {
            clip_model: get_comp("clip_service"),
            vector_db: get_comp("vector_db"),
            gpu_backend: get_comp("gpu_backend"),
        })
    } else {
        None
    };

    // Mock system health for now
    let system = SystemHealth {
        cpu_usage: 0.0,
        memory_usage: 0.0,
        disk_usage: 0.0,
    };

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            HealthCheckResponse {
                status: api_status,
                version: env!("CARGO_PKG_VERSION").to_string(),
                uptime: health.uptime_seconds,
                services,
                system,
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
}

/// Semantic text search
pub async fn text_search(
    State(state): State<AppState>,
    Json(request): Json<TextSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let top_k = request.top_k;
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
        )
            .into_response();
    }

    // Encode query text
    let embedding = match state.clip_service.encode_text(&request.query).await {
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
            )
                .into_response();
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
            )
                .into_response();
        }
    };

    // Filter by threshold
    let threshold = request.threshold.unwrap_or(0.0);
    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= threshold)
        .collect();

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    // Record metrics
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;
    let total_results = results.len();

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id.clone(),
                        content: r
                            .metadata
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("content unavailable")
                            .to_string(),
                        score: Some(r.similarity),
                        metadata: r.metadata,
                        indexed_at: chrono::Utc::now(),
                        content_type: ContentType::Text,
                    })
                    .collect(),
                total_results,
                query: request.query,
                config: SearchConfig {
                    top_k,
                    threshold,
                    search_method: request.search_method.unwrap_or(SearchMethod::Cosine),
                    search_mode: SearchMode::Text,
                },
                performance: SearchPerformance {
                    embedding_time_ms: 0,
                    search_time_ms: processing_time,
                    total_time_ms: processing_time,
                    method_used: "cosine".to_string(),
                },
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
}

/// Image search (using base64 image)
pub async fn image_search(
    State(state): State<AppState>,
    Json(request): Json<ImageSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let top_k = request.top_k;
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
        )
            .into_response();
    }

    // Decode base64 image
    let image_data = match general_purpose::STANDARD.decode(&request.image_b64) {
        Ok(data) => data,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InvalidRequest,
                    message: format!("Invalid base64 image data: {}", e),
                    details: None,
                    retry_after: None,
                }),
            )
                .into_response();
        }
    };

    // Encode image
    let embedding = match state.clip_service.encode_image(&image_data).await {
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
            )
                .into_response();
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
            )
                .into_response();
        }
    };

    // Filter by threshold
    let threshold = request.threshold.unwrap_or(0.0);
    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= threshold)
        .collect();

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    // Record metrics
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;
    let total_results = results.len();

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id.clone(),
                        content: r
                            .metadata
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("content unavailable")
                            .to_string(),
                        score: Some(r.similarity),
                        metadata: r.metadata,
                        indexed_at: chrono::Utc::now(),
                        content_type: ContentType::Image,
                    })
                    .collect(),
                total_results,
                query: format!("image_embedding_{}", embedding.len()),
                config: SearchConfig {
                    top_k,
                    threshold,
                    search_method: SearchMethod::Cosine,
                    search_mode: SearchMode::Image,
                },
                performance: SearchPerformance {
                    embedding_time_ms: 0,
                    search_time_ms: processing_time,
                    total_time_ms: processing_time,
                    method_used: "cosine".to_string(),
                },
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
}

/// Cross-modal search (text query, image results or vice versa)
pub async fn cross_modal_search(
    State(state): State<AppState>,
    Json(request): Json<CrossModalSearchRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let top_k = request.top_k;
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
        )
            .into_response();
    }

    // Generate embedding based on available query data
    let embedding = if !request.text_query.trim().is_empty() {
        // Use text query
        if request.text_query.trim().is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error_id: uuid::Uuid::new_v4().to_string(),
                    code: ErrorCode::InvalidRequest,
                    message: "Text query cannot be empty".to_string(),
                    details: None,
                    retry_after: None,
                }),
            )
                .into_response();
        }
        state.clip_service.encode_text(&request.text_query).await
    } else if let Some(image_b64) = &request.image_b64 {
        // Decode base64 image
        let image_data = match general_purpose::STANDARD.decode(image_b64) {
            Ok(data) => data,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error_id: uuid::Uuid::new_v4().to_string(),
                        code: ErrorCode::InvalidRequest,
                        message: format!("Invalid base64 image data: {}", e),
                        details: None,
                        retry_after: None,
                    }),
                )
                    .into_response();
            }
        };
        state.clip_service.encode_image(&image_data).await
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error_id: uuid::Uuid::new_v4().to_string(),
                code: ErrorCode::InvalidRequest,
                message: "Either text_query or image_b64 must be provided".to_string(),
                details: None,
                retry_after: None,
            }),
        )
            .into_response();
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
            )
                .into_response();
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
            )
                .into_response();
        }
    };

    // Filter by threshold
    let threshold = request.threshold.unwrap_or(0.0);
    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.similarity >= threshold)
        .collect();

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    // Record metrics
    #[allow(clippy::cast_precision_loss)]
    state.record_search(processing_time as f64).await;

    let health = state.health.read().await;
    let total_results = results.len();

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id.clone(),
                        content: r
                            .metadata
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("content unavailable")
                            .to_string(),
                        score: Some(r.similarity),
                        metadata: r.metadata,
                        indexed_at: chrono::Utc::now(),
                        content_type: ContentType::Multimodal,
                    })
                    .collect(),
                total_results,
                query: if request.text_query.is_empty() {
                    "cross_modal_image_query".to_string()
                } else {
                    request.text_query
                },
                config: SearchConfig {
                    top_k,
                    threshold,
                    search_method: SearchMethod::Cosine,
                    search_mode: SearchMode::CrossModal,
                },
                performance: SearchPerformance {
                    embedding_time_ms: 0,
                    search_time_ms: processing_time,
                    total_time_ms: processing_time,
                    method_used: "cosine".to_string(),
                },
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
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
        )
            .into_response();
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
        )
            .into_response();
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
            IndexContent::Image { data } => state.clip_service.encode_image(data).await,
        };

        let embedding = match embedding {
            Ok(emb) => emb,
            Err(e) => {
                errors.push(format!("Failed to encode item {}: {}", item.id, e));
                continue;
            }
        };

        // Add to vector database
        if let Err(e) = state
            .vector_db
            .add(item.id.clone(), embedding, serde_json::json!(item.metadata))
            .await
        {
            errors.push(format!("Failed to index item {}: {}", item.id, e));
            continue;
        }

        indexed_count += 1;
        state.record_index_operation().await;
    }

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            IndexResponse {
                batch_id: request.batch_id,
                indexed_count,
                failed_count: errors.len(),
                errors,
                processing_time_ms: processing_time,
            },
            ResponseMeta {
                processing_time_ms: processing_time,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
}

/// Get search performance benchmarks
pub async fn benchmark_search(
    State(state): State<AppState>,
    Query(params): Query<BenchmarkQuery>,
) -> impl IntoResponse {
    let start_time = chrono::Utc::now();
    let start_instant = Instant::now();

    // Generate test queries
    let num_queries = params.num_queries;
    let top_k = params.top_k;
    let test_queries = generate_test_queries(num_queries);

    let mut latencies = Vec::new();
    let mut error_count = 0;

    // Run benchmark queries
    for query in test_queries {
        let query_start = Instant::now();

        let Ok(embedding) = state.clip_service.encode_text(&query).await else {
            error_count += 1;
            continue;
        };

        let search_res = state.vector_db.search(&embedding, top_k).await;
        if search_res.is_err() {
            error_count += 1;
            continue;
        }

        let latency = query_start.elapsed().as_secs_f64() * 1000.0; // ms
        latencies.push(latency);
    }

    let duration_ms = u64::try_from(start_instant.elapsed().as_millis()).unwrap_or(u64::MAX);
    let total_queries = latencies.len() + error_count;

    let avg_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let p95_latency = percentile(&latencies, 95.0);
    let p99_latency = percentile(&latencies, 99.0);
    let qps = if duration_ms > 0 {
        (latencies.len() as f64) / (duration_ms as f64 / 1000.0)
    } else {
        0.0
    };

    let performance = BenchmarkPerformance {
        queries_per_second: qps,
        avg_latency_ms: avg_latency,
        p95_latency_ms: p95_latency,
        p99_latency_ms: p99_latency,
        error_count,
    };

    let summary = BenchmarkSummary {
        overall_qps: qps,
        total_queries,
        total_errors: error_count,
        duration_ms,
    };

    let system_config = SystemConfig {
        cpu_cores: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        memory_mb: 0, // Placeholder
        gpu_info: None,
        os: std::env::consts::OS.to_string(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let metadata = BenchmarkMetadata {
        start_time,
        end_time: chrono::Utc::now(),
        version: "1.0.0".to_string(),
        system_config,
    };

    let result = BenchmarkResult {
        query_type: params
            .query_types
            .first()
            .copied()
            .unwrap_or(BenchmarkQueryType::Text),
        top_k,
        performance,
    };

    let response = BenchmarkResponse {
        results: vec![result],
        summary,
        metadata,
    };

    let health = state.health.read().await;

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            response,
            ResponseMeta {
                processing_time_ms: duration_ms,
                api_version: "v1".to_string(),
                cache_hit: false,
                service_status: health.status.clone(),
            },
        )),
    )
        .into_response()
}

/// Get service metrics
pub async fn get_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let start_time = Instant::now();

    let metrics = state.get_metrics().await;
    let db_stats = (state.vector_db.stats().await).ok();

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
    let health = state.health.read().await;

    // Create default prometheus metrics string
    let prometheus_metrics = format!(
        "# HELP uptime_seconds Service uptime in seconds\n# TYPE uptime_seconds gauge\nuptime_seconds {}\n",
        health.uptime_seconds
    );

    (
        StatusCode::OK,
        Json(ApiResponse::new(
            uuid::Uuid::new_v4().to_string(),
            MetricsResponse {
                metrics: prometheus_metrics,
                collected_at: chrono::Utc::now(),
                format_version: "1.0".to_string(),
                search_metrics: serde_json::to_value(&metrics).unwrap_or_default(),
                database_stats: db_stats.map(|s| serde_json::to_value(&s).unwrap_or_default()),
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
        .into_response()
}

/// Generate test queries for benchmarking
fn generate_test_queries(count: usize) -> Vec<String> {
    let templates = [
        "a photo of a {}",
        "an image showing {}",
        "picture of {}",
        "beautiful {}",
        "close-up of {}",
    ];

    let objects = [
        "cat",
        "dog",
        "car",
        "house",
        "tree",
        "person",
        "bicycle",
        "bird",
        "flower",
        "mountain",
        "ocean",
        "sunset",
        "city",
        "building",
        "food",
        "animal",
        "vehicle",
        "landscape",
        "portrait",
        "nature",
    ];

    (0..count)
        .map(|i| {
            let template = &templates[i % templates.len()];
            let object = &objects[i % objects.len()];
            template.replace("{}", object)
        })
        .collect()
}

/// Calculate percentile from a sorted vector
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    let index = (p.clamp(0.0, 100.0) / 100.0 * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}
