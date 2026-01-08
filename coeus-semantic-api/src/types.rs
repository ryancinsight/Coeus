//! Request and Response Types for Semantic Search API
//!
//! Complete type definitions with comprehensive validation and serialization
//! for production-grade API interactions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Unique request identifier for correlation
    pub request_id: String,

    /// Response timestamp (ISO 8601)
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Response data
    pub data: T,

    /// Processing metadata
    pub meta: ResponseMeta,
}

impl<T> ApiResponse<T> {
    /// Create new API response
    pub fn new(request_id: String, data: T, meta: ResponseMeta) -> Self {
        Self {
            request_id,
            timestamp: chrono::Utc::now(),
            data,
            meta,
        }
    }
}

/// Response metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMeta {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// API version
    pub api_version: String,

    /// Cache hit status
    pub cache_hit: bool,

    /// Service status
    pub service_status: ServiceStatus,
}

/// Service operational status
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ServiceStatus {
    /// Service operational
    #[serde(rename = "operational")]
    Operational,

    /// Service degraded with issues
    #[serde(rename = "degraded")]
    Degraded,

    /// Service undergoing maintenance
    #[serde(rename = "maintenance")]
    Maintenance,
}

/// Generic API error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Unique error identifier
    pub error_id: String,

    /// Error code for programmatic handling
    pub code: ErrorCode,

    /// Human-readable error message
    pub message: String,

    /// Error details for debugging
    pub details: Option<HashMap<String, serde_json::Value>>,

    /// Retry information
    pub retry_after: Option<u32>,
}

/// Standardized error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    /// Invalid request input
    #[serde(rename = "invalid_request")]
    InvalidRequest,

    /// Requested resource not found
    #[serde(rename = "not_found")]
    NotFound,

    /// Service rate limit exceeded
    #[serde(rename = "rate_limited")]
    RateLimited,

    /// Internal service error
    #[serde(rename = "internal_error")]
    InternalError,

    /// Service temporarily unavailable
    #[serde(rename = "service_unavailable")]
    ServiceUnavailable,

    /// Authentication required
    #[serde(rename = "authentication_required")]
    AuthenticationRequired,

    /// Insufficient permissions
    #[serde(rename = "insufficient_permissions")]
    InsufficientPermissions,

    /// Authentication failed
    #[serde(rename = "unauthorized")]
    Unauthorized,

    /// Access forbidden
    #[serde(rename = "forbidden")]
    Forbidden,
}

/// Health check request
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckRequest {
    /// Include detailed service information
    pub detailed: Option<bool>,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    /// Overall health status
    pub status: HealthStatus,

    /// Service version
    pub version: String,

    /// Service uptime in seconds
    pub uptime: u64,

    /// Detailed service information
    pub services: Option<ServiceHealth>,

    /// System resources
    pub system: SystemHealth,
}

/// Overall health status
#[derive(Debug, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operational
    #[serde(rename = "healthy")]
    Healthy,

    /// Partial system degradation
    #[serde(rename = "degraded")]
    Degraded,

    /// Critical systems unavailable
    #[serde(rename = "unhealthy")]
    Unhealthy,
}

/// Detailed service health information
#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// CLIP model status
    pub clip_model: ComponentHealth,

    /// Vector database status
    pub vector_db: ComponentHealth,

    /// GPU backend status
    pub gpu_backend: ComponentHealth,
}

/// System resource health
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemHealth {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage percentage
    pub memory_usage: f32,

    /// Disk usage percentage
    pub disk_usage: f32,
}

/// Individual component health
#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: ComponentStatus,

    /// Status message
    pub message: String,

    /// Component version
    pub version: Option<String>,

    /// Last health check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// Component operational status
#[derive(Debug, Serialize, Deserialize)]
pub enum ComponentStatus {
    /// Component operational
    #[serde(rename = "operational")]
    Operational,

    /// Component impaired
    #[serde(rename = "impaired")]
    Impaired,

    /// Component unavailable
    #[serde(rename = "unavailable")]
    Unavailable,
}

/// Text search request
#[derive(Debug, Serialize, Deserialize)]
pub struct TextSearchRequest {
    /// Text query for semantic search
    pub query: String,

    /// Maximum number of results to return (1-100)
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity threshold (0.0-1.0)
    pub threshold: Option<f32>,

    /// Search method preference
    pub search_method: Option<SearchMethod>,

    /// Include similarity scores in response
    #[serde(default = "default_include_scores")]
    pub include_scores: bool,
}

/// Image search request (base64 encoded)
#[derive(Debug, Serialize, Deserialize)]
pub struct ImageSearchRequest {
    /// Base64-encoded image data
    pub image_b64: String,

    /// Image format (auto-detected if missing)
    pub format: Option<ImageFormat>,

    /// Maximum number of results to return (1-100)
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity threshold (0.0-1.0)
    pub threshold: Option<f32>,

    /// Include similarity scores in response
    #[serde(default = "default_include_scores")]
    pub include_scores: bool,
}

/// Cross-modal search request
#[derive(Debug, Serialize, Deserialize)]
pub struct CrossModalSearchRequest {
    /// Text query
    pub text_query: String,

    /// Optional image data (base64)
    pub image_b64: Option<String>,

    /// Image format (if image provided)
    pub format: Option<ImageFormat>,

    /// Maximum number of results to return (1-100)
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity threshold
    pub threshold: Option<f32>,

    /// Include similarity scores
    #[serde(default = "default_include_scores")]
    pub include_scores: bool,
}

/// Supported image formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImageFormat {
    #[serde(rename = "jpeg")]
    Jpeg,
    #[serde(rename = "png")]
    Png,
    #[serde(rename = "webp")]
    Webp,
}

/// Search method preference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchMethod {
    /// Standardized cosine similarity search
    #[serde(rename = "cosine")]
    Cosine,

    /// Euclidean distance search
    #[serde(rename = "euclidean")]
    Euclidean,

    /// Dot product search
    #[serde(rename = "dot_product")]
    DotProduct,
}

/// Search result item
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result identifier
    pub id: String,

    /// Result content/type description
    pub content: String,

    /// Similarity score (if requested)
    pub score: Option<f32>,

    /// Result metadata
    pub metadata: serde_json::Value,

    /// Timestamp when this result was indexed
    pub indexed_at: chrono::DateTime<chrono::Utc>,

    /// Content type
    pub content_type: ContentType,
}

/// Content classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "multimodal")]
    Multimodal,
}

/// Content indexing request
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexContentRequest {
    /// Text content to index
    pub texts: Option<Vec<IndexText>>,

    /// Image content to index (base64 encoded)
    pub images: Option<Vec<IndexImage>>,

    /// Batch processing options
    pub options: IndexOptions,
}

/// Text content for indexing
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexText {
    /// Unique identifier for the content
    pub id: String,

    /// Text content
    pub content: String,

    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Image content for indexing
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexImage {
    /// Unique identifier for the image
    pub id: String,

    /// Base64-encoded image data
    pub data_b64: String,

    /// Image format
    pub format: Option<ImageFormat>,

    /// Associated text caption/description
    pub caption: Option<String>,

    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Indexing options
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexOptions {
    /// Whether to overwrite existing content with same ID
    #[serde(default = "default_overwrite")]
    pub overwrite: bool,

    /// Batch size for processing
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Skip validation (use with caution)
    #[serde(default)]
    pub skip_validation: bool,
}

/// Generic search response
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,

    /// Total number of results found
    pub total_results: usize,

    /// Search query used
    pub query: String,

    /// Search configuration used
    pub config: SearchConfig,

    /// Search performance metrics
    pub performance: SearchPerformance,
}

/// Search configuration used
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Top-k value
    pub top_k: usize,

    /// Similarity threshold applied
    pub threshold: f32,

    /// Search method used
    pub search_method: SearchMethod,

    /// Search mode
    pub search_mode: SearchMode,
}

/// Search mode
#[derive(Debug, Serialize, Deserialize)]
pub enum SearchMode {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "cross_modal")]
    CrossModal,
}

/// Content types for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: Vec<u8> },
}

/// Index item for batch indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexItem {
    /// Unique identifier for the item
    pub id: String,
    /// Content to index
    pub content: IndexContent,
    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Index request for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    /// Items to index
    pub items: Vec<IndexItem>,
    /// Optional batch identifier
    pub batch_id: Option<String>,
}

/// Index response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    /// Batch identifier (if provided in request)
    pub batch_id: Option<String>,
    /// Number of items successfully indexed
    pub indexed_count: usize,
    /// Number of items that failed to index
    pub failed_count: usize,
    /// Error messages for failed items
    pub errors: Vec<String>,
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Search performance metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchPerformance {
    /// Embedding computation time (ms)
    pub embedding_time_ms: u64,

    /// Vector search time (ms)
    pub search_time_ms: u64,

    /// Total processing time (ms)
    pub total_time_ms: u64,

    /// Search method used
    pub method_used: String,
}

/// Benchmark request
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    /// Number of queries to benchmark
    #[serde(default = "default_benchmark_queries")]
    pub num_queries: usize,

    /// Query types to include
    pub query_types: Vec<BenchmarkQueryType>,

    /// Benchmark configuration
    pub config: BenchmarkConfig,
}

/// Benchmark query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuery {
    /// Number of queries to run (1-100)
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
    /// Top-k value for searches (1-100)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Query types to include
    #[serde(default)]
    pub query_types: Vec<BenchmarkQueryType>,
}

/// Benchmark query types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkQueryType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "mixed")]
    Mixed,
}

/// Benchmark configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Top-k values to test
    pub top_k_values: Vec<usize>,

    /// Parallel request load
    #[serde(default)]
    pub concurrent_requests: usize,

    /// Duration in seconds to limit benchmark
    pub duration_limit_secs: Option<u64>,
}

/// Benchmark response
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    /// Benchmark results
    pub results: Vec<BenchmarkResult>,

    /// Overall benchmark summary
    pub summary: BenchmarkSummary,

    /// Benchmark metadata
    pub metadata: BenchmarkMetadata,
}

/// Individual benchmark result
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Query type benchmarked
    pub query_type: BenchmarkQueryType,

    /// Top-k value used
    pub top_k: usize,

    /// Performance metrics
    pub performance: BenchmarkPerformance,
}

/// Benchmark performance metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkPerformance {
    /// Queries per second
    pub queries_per_second: f64,

    /// Average latency (ms)
    pub avg_latency_ms: f64,

    /// 95th percentile latency (ms)
    pub p95_latency_ms: f64,

    /// 99th percentile latency (ms)
    pub p99_latency_ms: f64,

    /// Error count
    pub error_count: usize,
}

/// Benchmark summary
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Overall queries per second
    pub overall_qps: f64,

    /// Total queries processed
    pub total_queries: usize,

    /// Total errors encountered
    pub total_errors: usize,

    /// Benchmark duration (ms)
    pub duration_ms: u64,
}

/// Benchmark metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Benchmark start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Benchmark end time
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Benchmark version
    pub version: String,

    /// System configuration used
    pub system_config: SystemConfig,
}

/// System configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemConfig {
    /// CPU cores available
    pub cpu_cores: usize,

    /// Memory available (MB)
    pub memory_mb: usize,

    /// GPU information
    pub gpu_info: Option<String>,

    /// Operating system
    pub os: String,

    /// Application version
    pub app_version: String,
}

/// Metrics response
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// Prometheus metric format
    pub metrics: String,

    /// Metrics collection timestamp
    pub collected_at: chrono::DateTime<chrono::Utc>,

    /// Metrics format version
    pub format_version: String,

    /// Search-related metrics
    pub search_metrics: serde_json::Value,

    /// Database statistics
    pub database_stats: Option<serde_json::Value>,

    /// Service uptime in seconds
    pub uptime_seconds: u64,
}

// Validation defaults
fn default_top_k() -> usize {
    10
}
fn default_include_scores() -> bool {
    true
}
fn default_overwrite() -> bool {
    false
}
fn default_batch_size() -> usize {
    32
}
fn default_benchmark_queries() -> usize {
    1000
}
fn default_num_queries() -> usize {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_response_serialization() -> Result<(), serde_json::Error> {
        let meta = ResponseMeta {
            processing_time_ms: 150,
            api_version: "1.0".to_string(),
            cache_hit: false,
            service_status: ServiceStatus::Operational,
        };

        let response = ApiResponse::new("req-123".to_string(), "test data".to_string(), meta);

        let json = serde_json::to_string(&response)?;
        let deserialized: ApiResponse<String> = serde_json::from_str(&json)?;

        assert_eq!(deserialized.request_id, "req-123");
        assert_eq!(deserialized.data, "test data");

        Ok(())
    }

    #[test]
    fn test_text_search_request_validation() {
        let request = TextSearchRequest {
            query: "test query".to_string(),
            top_k: 50,
            threshold: Some(0.7),
            search_method: Some(SearchMethod::Cosine),
            include_scores: true,
        };

        assert_eq!(request.query, "test query");
        assert_eq!(request.top_k, 50);
        assert!(matches!(
            request.threshold,
            Some(threshold) if (threshold - 0.7).abs() < 1e-12
        ));
    }

    #[test]
    fn test_error_response_serialization() -> Result<(), serde_json::Error> {
        let error = ErrorResponse {
            error_id: "err-456".to_string(),
            code: ErrorCode::InvalidRequest,
            message: "Invalid input provided".to_string(),
            details: Some(HashMap::from([
                ("field".to_string(), serde_json::json!("query")),
                ("reason".to_string(), serde_json::json!("empty")),
            ])),
            retry_after: None,
        };

        let json = serde_json::to_string(&error)?;
        let deserialized: ErrorResponse = serde_json::from_str(&json)?;

        assert_eq!(deserialized.code, ErrorCode::InvalidRequest);
        assert_eq!(deserialized.message, "Invalid input provided");

        Ok(())
    }
}
