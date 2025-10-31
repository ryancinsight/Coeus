//! Application State Management for Semantic Search API
//!
//! Production-grade state management with proper resource lifecycle,
//! health monitoring, and enterprise features.

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::*;

/// Application state container with all services and configuration
#[derive(Clone)]
pub struct AppState {
    /// CLIP model service for embedding generation
    pub clip_service: Arc<dyn CLIPService + Send + Sync>,

    /// Vector database for similarity search
    pub vector_db: Arc<dyn VectorDatabase + Send + Sync>,

    /// Search metrics and performance monitoring
    pub metrics: Arc<RwLock<SearchMetrics>>,

    /// Configuration settings
    pub config: AppConfig,

    /// Health status tracker
    pub health: Arc<RwLock<HealthStatus>>,
}

/// CLIP service trait for embedding generation
#[async_trait::async_trait]
pub trait CLIPService: Send + Sync {
    /// Encode text into embedding vector
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>, SemanticError>;

    /// Encode image data into embedding vector
    async fn encode_image(&self, image_data: &[u8]) -> Result<Vec<f32>, SemanticError>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get service health status
    async fn health_check(&self) -> Result<(), SemanticError>;
}

/// Vector database trait for similarity search
#[async_trait::async_trait]
pub trait VectorDatabase: Send + Sync {
    /// Add embedding with metadata
    async fn add(&self, id: String, embedding: Vec<f32>, metadata: serde_json::Value) -> Result<(), SemanticError>;

    /// Search for similar embeddings
    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>, SemanticError>;

    /// Delete embedding by ID
    async fn delete(&self, id: &str) -> Result<bool, SemanticError>;

    /// Get database statistics
    async fn stats(&self) -> Result<DatabaseStats, SemanticError>;

    /// Clear all embeddings
    async fn clear(&self) -> Result<(), SemanticError>;
}

/// Search result with similarity score and metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// Unique identifier
    pub id: String,

    /// Cosine similarity score (0.0 to 1.0)
    pub similarity: f32,

    /// Associated metadata
    pub metadata: serde_json::Value,
}

/// Database statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatabaseStats {
    /// Total number of indexed items
    pub total_items: usize,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Index size in bytes
    pub index_size_bytes: usize,

    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Search performance metrics
#[derive(Debug, Clone, Default)]
pub struct SearchMetrics {
    /// Total search requests processed
    pub total_requests: u64,

    /// Average search latency in milliseconds
    pub avg_search_latency_ms: f64,

    /// Total index operations
    pub total_index_operations: u64,

    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,

    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Peak memory usage in bytes
    pub peak_memory_usage_bytes: usize,
}

/// Application configuration
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Maximum batch size for processing
    pub max_batch_size: usize,

    /// Default number of results to return
    pub default_top_k: usize,

    /// Maximum number of results to return
    pub max_top_k: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,

    /// Rate limiting: requests per minute per IP
    pub rate_limit_per_minute: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            default_top_k: 10,
            max_top_k: 100,
            request_timeout_secs: 30,
            enable_caching: true,
            cache_ttl_secs: 300, // 5 minutes
            rate_limit_per_minute: 100,
        }
    }
}

/// Health status for service monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    /// Overall service status
    pub status: ServiceStatus,

    /// Component health checks
    pub components: HashMap<String, ComponentHealth>,

    /// Last health check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,

    /// Service uptime in seconds
    pub uptime_seconds: u64,
}

/// Individual component health
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: ComponentStatus,

    /// Health check message
    pub message: String,

    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
}

/// Component status enumeration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ComponentStatus {
    #[serde(rename = "healthy")]
    Healthy,

    #[serde(rename = "degraded")]
    Degraded,

    #[serde(rename = "unhealthy")]
    Unhealthy,
}

impl AppState {
    /// Create new application state with services
    pub fn new(
        clip_service: Arc<dyn CLIPService + Send + Sync>,
        vector_db: Arc<dyn VectorDatabase + Send + Sync>,
        config: AppConfig,
    ) -> Self {
        Self {
            clip_service,
            vector_db,
            metrics: Arc::new(RwLock::new(SearchMetrics::default())),
            config,
            health: Arc::new(RwLock::new(HealthStatus {
                status: ServiceStatus::Operational,
                components: HashMap::new(),
                last_check: chrono::Utc::now(),
                uptime_seconds: 0,
            })),
        }
    }

    /// Create test state for unit testing
    #[cfg(test)]
    pub fn new_for_testing() -> Result<Self, SemanticError> {
        use crate::mock_services::*;

        let clip_service = Arc::new(MockCLIPService::new());
        let vector_db = Arc::new(MockVectorDatabase::new());
        let config = AppConfig::default();

        Ok(Self::new(clip_service, vector_db, config))
    }

    /// Update health status
    pub async fn update_health(&self) -> Result<(), SemanticError> {
        let mut health = self.health.write().await;

        // Check CLIP service health
        let clip_health = match self.clip_service.health_check().await {
            Ok(_) => ComponentHealth {
                status: ComponentStatus::Healthy,
                message: "CLIP service operational".to_string(),
                response_time_ms: None,
            },
            Err(e) => ComponentHealth {
                status: ComponentStatus::Unhealthy,
                message: format!("CLIP service error: {}", e),
                response_time_ms: None,
            },
        };

        // Check vector database health
        let db_health = match self.vector_db.stats().await {
            Ok(_) => ComponentHealth {
                status: ComponentStatus::Healthy,
                message: "Vector database operational".to_string(),
                response_time_ms: None,
            },
            Err(e) => ComponentHealth {
                status: ComponentStatus::Unhealthy,
                message: format!("Database error: {}", e),
                response_time_ms: None,
            },
        };

        let mut components = HashMap::new();
        components.insert("clip_service".to_string(), clip_health);
        components.insert("vector_database".to_string(), db_health);

        // Determine overall status
        let overall_status = if components.values().all(|c| matches!(c.status, ComponentStatus::Healthy)) {
            ServiceStatus::Operational
        } else if components.values().any(|c| matches!(c.status, ComponentStatus::Unhealthy)) {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Maintenance
        };

        health.status = overall_status;
        health.components = components;
        health.last_check = chrono::Utc::now();

        Ok(())
    }

    /// Record search metrics
    pub async fn record_search(&self, latency_ms: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;

        // Update rolling average
        let alpha = 0.1; // Smoothing factor
        metrics.avg_search_latency_ms = metrics.avg_search_latency_ms * (1.0 - alpha) + latency_ms * alpha;
    }

    /// Record index operation
    pub async fn record_index_operation(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.total_index_operations += 1;
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> SearchMetrics {
        self.metrics.read().await.clone()
    }
}

#[cfg(test)]
mod mock_services {
    use super::*;
    use std::collections::HashMap;
    use tokio::sync::RwLock;

    pub struct MockCLIPService {
        embedding_dim: usize,
    }

    impl MockCLIPService {
        pub fn new() -> Self {
            Self { embedding_dim: 512 }
        }
    }

    #[async_trait::async_trait]
    impl CLIPService for MockCLIPService {
        async fn encode_text(&self, text: &str) -> Result<Vec<f32>, SemanticError> {
            let mut embedding = Vec::new();
            for i in 0..self.embedding_dim {
                let hash = text.chars().map(|c| c as u32).sum::<u32>() + i as u32;
                embedding.push((hash % 1000) as f32 / 1000.0);
            }
            Ok(embedding)
        }

        async fn encode_image(&self, image_data: &[u8]) -> Result<Vec<f32>, SemanticError> {
            let mut embedding = Vec::new();
            for i in 0..self.embedding_dim {
                let hash = image_data.len() as u32 + i as u32;
                embedding.push((hash % 1000) as f32 / 1000.0);
            }
            Ok(embedding)
        }

        fn embedding_dim(&self) -> usize {
            self.embedding_dim
        }

        async fn health_check(&self) -> Result<(), SemanticError> {
            Ok(())
        }
    }

    pub struct MockVectorDatabase {
        entries: Arc<RwLock<HashMap<String, (Vec<f32>, serde_json::Value)>>>,
    }

    impl MockVectorDatabase {
        pub fn new() -> Self {
            Self {
                entries: Arc::new(RwLock::new(HashMap::new())),
            }
        }
    }

    #[async_trait::async_trait]
    impl VectorDatabase for MockVectorDatabase {
        async fn add(&self, id: String, embedding: Vec<f32>, metadata: serde_json::Value) -> Result<(), SemanticError> {
            let mut entries = self.entries.write().await;
            entries.insert(id, (embedding, metadata));
            Ok(())
        }

        async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>, SemanticError> {
            let entries = self.entries.read().await;
            let mut results: Vec<_> = entries.iter()
                .map(|(id, (emb, meta))| {
                    let similarity = cosine_similarity(query_embedding, emb);
                    SearchResult {
                        id: id.clone(),
                        similarity,
                        metadata: meta.clone(),
                    }
                })
                .collect();

            results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            Ok(results.into_iter().take(top_k).collect())
        }

        async fn delete(&self, id: &str) -> Result<bool, SemanticError> {
            let mut entries = self.entries.write().await;
            Ok(entries.remove(id).is_some())
        }

        async fn stats(&self) -> Result<DatabaseStats, SemanticError> {
            let entries = self.entries.read().await;
            Ok(DatabaseStats {
                total_items: entries.len(),
                embedding_dim: entries.values().next().map(|(emb, _)| emb.len()).unwrap_or(512),
                index_size_bytes: entries.len() * 512 * 4, // Rough estimate
                last_updated: chrono::Utc::now(),
            })
        }

        async fn clear(&self) -> Result<(), SemanticError> {
            let mut entries = self.entries.write().await;
            entries.clear();
            Ok(())
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}





