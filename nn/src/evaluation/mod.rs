//! CLIP Model Evaluation Framework
//!
//! Comprehensive evaluation suite for CLIP models including:
//! - Retrieval evaluation (text-to-image, image-to-text)
//! - Zero-shot classification on standard benchmarks
//! - Embedding space quality analysis
//! - Performance profiling and benchmarking
//!
//! This module provides production-ready evaluation tools for systematic
//! CLIP model assessment and benchmarking.

pub mod benchmark;
pub mod embeddings;
pub mod profiling;
pub mod retrieval;
pub mod zeroshot;

// Re-exports for convenient access
pub use benchmark::*;
pub use embeddings::*;
pub use profiling::*;
pub use retrieval::*;
pub use zeroshot::*;

/// Common evaluation dataset interface
pub trait EvaluationDataset {
    /// Get dataset name
    fn name(&self) -> &str;

    /// Get number of samples
    fn len(&self) -> usize;

    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get sample by index
    fn get_sample(&self, index: usize) -> Option<&dyn std::any::Any>;

    /// Get image embeddings (for evaluation)
    fn image_embeddings(&self) -> &[Vec<f32>];

    /// Get text embeddings (for evaluation)
    fn text_embeddings(&self) -> &[Vec<f32>];
}

/// Evaluation metrics aggregation
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Confidence interval (optional)
    pub confidence_interval: Option<(f64, f64)>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl EvaluationMetrics {
    /// Create new evaluation metrics
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            confidence_interval: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add confidence interval
    pub fn with_confidence_interval(mut self, lower: f64, upper: f64) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Evaluation result aggregator
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Dataset name
    pub dataset_name: String,
    /// Model name/version
    pub model_name: String,
    /// Evaluation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// All computed metrics
    pub metrics: Vec<EvaluationMetrics>,
    /// Summary statistics
    pub summary: std::collections::HashMap<String, f64>,
}

impl EvaluationResult {
    /// Create new evaluation result
    pub fn new(dataset_name: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            dataset_name: dataset_name.into(),
            model_name: model_name.into(),
            timestamp: chrono::Utc::now(),
            metrics: Vec::new(),
            summary: std::collections::HashMap::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, metric: EvaluationMetrics) {
        self.metrics.push(metric);
    }

    /// Add summary statistic
    pub fn add_summary(&mut self, key: impl Into<String>, value: f64) {
        self.summary.insert(key.into(), value);
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&EvaluationMetrics> {
        self.metrics.iter().find(|m| m.name == name)
    }

    /// Get all metrics with a given prefix
    pub fn get_metrics_with_prefix(&self, prefix: &str) -> Vec<&EvaluationMetrics> {
        self.metrics
            .iter()
            .filter(|m| m.name.starts_with(prefix))
            .collect()
    }

    /// Generate human-readable report
    pub fn generate_report(&self) -> String {
        let mut report = "📊 CLIP Evaluation Report\n".to_string();
        report.push_str(&format!("Dataset: {}\n", self.dataset_name));
        report.push_str(&format!("Model: {}\n", self.model_name));
        report.push_str(&format!(
            "Timestamp: {}\n",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str("\n📈 Metrics:\n");

        for metric in &self.metrics {
            report.push_str(&format!("  {}: {:.4}", metric.name, metric.value));
            if let Some((lower, upper)) = metric.confidence_interval {
                report.push_str(&format!(" [{:.4}, {:.4}]", lower, upper));
            }
            report.push('\n');
        }

        if !self.summary.is_empty() {
            report.push_str("\n📋 Summary:\n");
            for (key, value) in &self.summary {
                report.push_str(&format!("  {}: {:.4}\n", key, value));
            }
        }

        report
    }
}

/// Configuration for evaluation runs
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Number of workers for parallel evaluation
    pub num_workers: usize,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level (0.95 for 95% CI)
    pub confidence_level: f64,
    /// Subsample size for faster evaluation (None = use all data)
    pub subsample_size: Option<usize>,
    /// Random seed for reproducible subsampling
    pub random_seed: u64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            subsample_size: None,
            random_seed: 42,
        }
    }
}

/// CLIP evaluation configuration
#[derive(Debug, Clone)]
pub struct ClipEvaluationConfig {
    /// Base evaluation configuration
    pub base_config: EvaluationConfig,
    /// Image size for evaluation
    pub image_size: usize,
    /// Maximum sequence length for text
    pub max_seq_length: usize,
    /// Whether to use cached embeddings
    pub use_cached_embeddings: bool,
    /// Cache directory for embeddings
    pub cache_dir: std::path::PathBuf,
    /// Top-k values for retrieval evaluation
    pub retrieval_top_k: Vec<usize>,
    /// Batch size for evaluation
    pub eval_batch_size: usize,
}

impl Default for ClipEvaluationConfig {
    fn default() -> Self {
        Self {
            base_config: EvaluationConfig::default(),
            image_size: 224,
            max_seq_length: 77,
            use_cached_embeddings: true,
            cache_dir: std::path::PathBuf::from("./clip_cache"),
            retrieval_top_k: vec![1, 5, 10],
            eval_batch_size: 32,
        }
    }
}

/// Benchmark dataset interface
pub trait BenchmarkDataset: EvaluationDataset {
    /// Get class names for classification
    fn class_names(&self) -> Vec<String>;

    /// Get number of classes
    fn num_classes(&self) -> usize {
        self.class_names().len()
    }

    /// Get image and label pairs for evaluation
    fn get_image_label_pairs(&self) -> Vec<(Vec<u8>, usize)>;

    /// Get labels for evaluation
    fn labels(&self) -> &[usize];
}

/// CLIP model evaluator trait
pub trait ClipModelEvaluator {
    /// Generate image embeddings
    fn encode_images(&self, images: &[Vec<u8>]) -> crate::core::error::Result<Vec<Vec<f32>>>;

    /// Generate text embeddings
    fn encode_texts(&self, texts: &[String]) -> crate::core::error::Result<Vec<Vec<f32>>>;

    /// Compute similarity between image and text embeddings
    fn compute_similarity(
        &self,
        image_embeddings: &[Vec<f32>],
        text_embeddings: &[Vec<f32>],
    ) -> crate::core::error::Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_metrics() {
        let metric = EvaluationMetrics::new("test_accuracy", 0.85)
            .with_confidence_interval(0.82, 0.88)
            .with_metadata("dataset", "test");

        assert_eq!(metric.name, "test_accuracy");
        assert!((metric.value - 0.85).abs() < 1e-6);
        assert_eq!(metric.confidence_interval, Some((0.82, 0.88)));
        assert_eq!(metric.metadata.get("dataset"), Some(&"test".to_string()));
    }

    #[test]
    fn test_evaluation_result() {
        let mut result = EvaluationResult::new("test_dataset", "test_model");

        let metric = EvaluationMetrics::new("accuracy", 0.90);
        result.add_metric(metric);
        result.add_summary("avg_score", 0.88);

        assert_eq!(result.dataset_name, "test_dataset");
        assert_eq!(result.model_name, "test_model");
        assert_eq!(result.metrics.len(), 1);
        assert_eq!(result.summary.len(), 1);

        let found_metric = result.get_metric("accuracy");
        assert!(found_metric.is_some());
        assert_eq!(found_metric.unwrap().value, 0.90);
    }

    #[test]
    fn test_evaluation_config_default() {
        let config = EvaluationConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_workers, 4);
        assert!(!config.compute_confidence_intervals);
        assert!((config.confidence_level - 0.95).abs() < 1e-6);
        assert!(config.subsample_size.is_none());
        assert_eq!(config.random_seed, 42);
    }

    #[test]
    fn test_clip_evaluation_config_default() {
        let config = ClipEvaluationConfig::default();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.max_seq_length, 77);
        assert!(config.use_cached_embeddings);
        assert_eq!(config.cache_dir, std::path::PathBuf::from("./clip_cache"));
    }
}
