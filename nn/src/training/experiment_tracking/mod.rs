//! Experiment Tracking and Reproducibility Framework
//!
//! Comprehensive experiment tracking for machine learning research,
//! enabling reproducible experiments, hyperparameter optimization,
//! and collaboration across research teams.

use crate::core::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Experiment specification defining what to run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    /// Unique experiment identifier
    pub id: String,
    /// Human-readable experiment name
    pub name: String,
    /// Experiment description
    pub description: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Experiment tags for organization
    pub tags: Vec<String>,
    /// Configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    /// Code version/commit hash
    pub code_version: String,
    /// Environment information
    pub environment: EnvironmentInfo,
}

/// Experiment execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Experiment specification
    pub spec: ExperimentSpec,
    /// Execution status
    pub status: ExperimentStatus,
    /// Start timestamp
    pub started_at: Option<u64>,
    /// Completion timestamp
    pub completed_at: Option<u64>,
    /// Metrics collected during execution
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    /// Artifacts produced (model checkpoints, logs, etc.)
    pub artifacts: Vec<Artifact>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Experiment execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Metric point with timestamp and value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Step number (e.g., training step)
    pub step: u64,
    /// Timestamp when metric was recorded
    pub timestamp: u64,
}

/// Artifact produced during experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// File path or storage location
    pub path: String,
    /// File size in bytes
    pub size_bytes: Option<u64>,
    /// Metadata about the artifact
    pub metadata: HashMap<String, String>,
}

/// Artifact types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    ModelCheckpoint,
    TrainingLog,
    EvaluationResult,
    Configuration,
    DatasetSample,
    Visualization,
}

/// Environment information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu: String,
    /// GPU information (if available)
    pub gpu: Option<String>,
    /// RAM size in GB
    pub ram_gb: usize,
    /// Python version (if applicable)
    pub python_version: Option<String>,
    /// Rust version
    pub rust_version: String,
    /// CUDA version (if applicable)
    pub cuda_version: Option<String>,
    /// Installed packages/dependencies
    pub dependencies: HashMap<String, String>,
}

/// Experiment tracker for managing experiment lifecycle
#[derive(Debug)]
pub struct ExperimentTracker {
    /// Experiment specification
    spec: ExperimentSpec,
    /// Current status
    status: ExperimentStatus,
    /// Start time
    started_at: Option<u64>,
    /// Metrics collected
    metrics: HashMap<String, Vec<MetricPoint>>,
    /// Artifacts produced
    artifacts: Vec<Artifact>,
    /// Storage backend for persistence
    storage: Arc<dyn ExperimentStorage>,
}

/// Experiment storage trait for persistence
#[async_trait::async_trait]
pub trait ExperimentStorage: Send + Sync + std::fmt::Debug {
    /// Save experiment result
    async fn save_experiment(&self, result: &ExperimentResult) -> Result<()>;

    /// Load experiment result by ID
    async fn load_experiment(&self, id: &str) -> Result<Option<ExperimentResult>>;

    /// List experiments with optional filtering
    async fn list_experiments(
        &self,
        filter: Option<&ExperimentFilter>,
    ) -> Result<Vec<ExperimentResult>>;

    /// Save artifact to storage
    async fn save_artifact(
        &self,
        experiment_id: &str,
        artifact: &Artifact,
        data: &[u8],
    ) -> Result<String>;

    /// Load artifact from storage
    async fn load_artifact(
        &self,
        experiment_id: &str,
        artifact_name: &str,
    ) -> Result<Option<Vec<u8>>>;
}

/// Filter for experiment queries
#[derive(Debug, Clone, Default)]
pub struct ExperimentFilter {
    /// Filter by name pattern
    pub name_pattern: Option<String>,
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    /// Filter by status
    pub status: Option<ExperimentStatus>,
    /// Filter by creation time range
    pub created_after: Option<u64>,
    pub created_before: Option<u64>,
}

/// In-memory experiment storage for development/testing
#[derive(Debug)]
pub struct InMemoryStorage {
    experiments: RwLock<HashMap<String, ExperimentResult>>,
    artifacts: RwLock<HashMap<String, HashMap<String, Vec<u8>>>>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self {
            experiments: RwLock::new(HashMap::new()),
            artifacts: RwLock::new(HashMap::new()),
        }
    }
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl ExperimentStorage for InMemoryStorage {
    async fn save_experiment(&self, result: &ExperimentResult) -> Result<()> {
        let mut experiments = self.experiments.write().await;
        experiments.insert(result.spec.id.clone(), result.clone());
        Ok(())
    }

    async fn load_experiment(&self, id: &str) -> Result<Option<ExperimentResult>> {
        let experiments = self.experiments.read().await;
        Ok(experiments.get(id).cloned())
    }

    async fn list_experiments(
        &self,
        filter: Option<&ExperimentFilter>,
    ) -> Result<Vec<ExperimentResult>> {
        let experiments = self.experiments.read().await;
        let mut results: Vec<_> = experiments.values().cloned().collect();

        if let Some(filter) = filter {
            results.retain(|exp| {
                // Apply filters
                if let Some(pattern) = &filter.name_pattern {
                    if !exp.spec.name.contains(pattern) {
                        return false;
                    }
                }

                if let Some(req_tags) = &filter.tags {
                    if !req_tags.iter().all(|tag| exp.spec.tags.contains(tag)) {
                        return false;
                    }
                }

                if let Some(status) = &filter.status {
                    if &exp.status != status {
                        return false;
                    }
                }

                if let Some(after) = filter.created_after {
                    if exp.spec.created_at < after {
                        return false;
                    }
                }

                if let Some(before) = filter.created_before {
                    if exp.spec.created_at > before {
                        return false;
                    }
                }

                true
            });
        }

        Ok(results)
    }

    async fn save_artifact(
        &self,
        experiment_id: &str,
        artifact: &Artifact,
        data: &[u8],
    ) -> Result<String> {
        let mut artifacts = self.artifacts.write().await;
        let exp_artifacts = artifacts
            .entry(experiment_id.to_string())
            .or_insert_with(HashMap::new);
        exp_artifacts.insert(artifact.name.clone(), data.to_vec());

        Ok(format!("memory://{}/{}", experiment_id, artifact.name))
    }

    async fn load_artifact(
        &self,
        experiment_id: &str,
        artifact_name: &str,
    ) -> Result<Option<Vec<u8>>> {
        let artifacts = self.artifacts.read().await;
        Ok(artifacts
            .get(experiment_id)
            .and_then(|exp_artifacts| exp_artifacts.get(artifact_name))
            .cloned())
    }
}

impl ExperimentTracker {
    /// Create new experiment tracker
    pub fn new(spec: ExperimentSpec, storage: Arc<dyn ExperimentStorage>) -> Self {
        Self {
            spec,
            status: ExperimentStatus::Pending,
            started_at: None,
            metrics: HashMap::new(),
            artifacts: Vec::new(),
            storage,
        }
    }

    /// Start the experiment
    pub async fn start(&mut self) -> Result<()> {
        self.status = ExperimentStatus::Running;
        self.started_at = Some(current_timestamp());

        // Log start event
        tracing::info!("Started experiment: {} ({})", self.spec.name, self.spec.id);

        Ok(())
    }

    /// Complete the experiment successfully
    pub async fn complete(&mut self) -> Result<()> {
        self.status = ExperimentStatus::Completed;
        let completed_at = current_timestamp();

        // Create result
        let result = ExperimentResult {
            spec: self.spec.clone(),
            status: self.status.clone(),
            started_at: self.started_at,
            completed_at: Some(completed_at),
            metrics: self.metrics.clone(),
            artifacts: self.artifacts.clone(),
            error_message: None,
        };

        // Save to storage
        self.storage.save_experiment(&result).await?;

        // Log completion
        tracing::info!(
            "Completed experiment: {} ({})",
            self.spec.name,
            self.spec.id
        );

        Ok(())
    }

    /// Fail the experiment with error message
    pub async fn fail(&mut self, error: &str) -> Result<()> {
        self.status = ExperimentStatus::Failed;
        let completed_at = current_timestamp();

        let result = ExperimentResult {
            spec: self.spec.clone(),
            status: self.status.clone(),
            started_at: self.started_at,
            completed_at: Some(completed_at),
            metrics: self.metrics.clone(),
            artifacts: self.artifacts.clone(),
            error_message: Some(error.to_string()),
        };

        self.storage.save_experiment(&result).await?;

        tracing::error!(
            "Failed experiment: {} ({}): {}",
            self.spec.name,
            self.spec.id,
            error
        );

        Ok(())
    }

    /// Record a metric value
    pub async fn record_metric(&mut self, name: &str, value: f64, step: u64) {
        let point = MetricPoint {
            name: name.to_string(),
            value,
            step,
            timestamp: current_timestamp(),
        };

        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(point);
    }

    /// Log hyperparameter value
    pub async fn log_hyperparameter(&mut self, name: &str, value: serde_json::Value) {
        self.spec.config.insert(name.to_string(), value);
    }

    /// Add artifact to experiment
    pub async fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Save artifact data
    pub async fn save_artifact_data(&self, artifact: &Artifact, data: &[u8]) -> Result<String> {
        self.storage
            .save_artifact(&self.spec.id, artifact, data)
            .await
    }

    /// Get current experiment status
    pub fn status(&self) -> &ExperimentStatus {
        &self.status
    }

    /// Get experiment ID
    pub fn id(&self) -> &str {
        &self.spec.id
    }
}

/// Utility function to create experiment specification
pub fn create_experiment_spec(
    name: String,
    description: String,
    tags: Vec<String>,
    config: HashMap<String, serde_json::Value>,
) -> ExperimentSpec {
    let id = format!("exp_{}", uuid::Uuid::new_v4().simple());

    ExperimentSpec {
        id,
        name,
        description,
        created_at: current_timestamp(),
        tags,
        config,
        code_version: env!("CARGO_PKG_VERSION").to_string(),
        environment: collect_environment_info(),
    }
}

/// Collect current environment information
fn collect_environment_info() -> EnvironmentInfo {
    EnvironmentInfo {
        os: std::env::consts::OS.to_string(),
        cpu: "Unknown".to_string(), // Would need external crate for CPU detection
        gpu: None,                  // Would need GPU detection
        ram_gb: 16,                 // Default assumption
        python_version: None,
        rust_version: rustc_version::version()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
        cuda_version: None,
        dependencies: HashMap::new(), // Would need to collect actual dependencies
    }
}

/// Get current timestamp in seconds since Unix epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_experiment_tracker_lifecycle() {
        let spec = create_experiment_spec(
            "test_experiment".to_string(),
            "Test experiment".to_string(),
            vec!["test".to_string()],
            HashMap::new(),
        );

        let storage = Arc::new(InMemoryStorage::new());
        let mut tracker = ExperimentTracker::new(spec, storage.clone());

        // Start experiment
        tracker.start().await.unwrap();
        assert!(matches!(tracker.status(), ExperimentStatus::Running));

        // Record metrics
        tracker.record_metric("loss", 0.5, 1).await;
        tracker.record_metric("accuracy", 0.85, 1).await;

        // Complete experiment
        tracker.complete().await.unwrap();
        assert!(matches!(tracker.status(), ExperimentStatus::Completed));

        // Verify experiment was saved
        let saved = storage
            .load_experiment(tracker.id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.spec.name, "test_experiment");
        assert!(saved.metrics.contains_key("loss"));
        assert!(saved.metrics.contains_key("accuracy"));
    }

    #[tokio::test]
    async fn test_experiment_filtering() {
        let storage = Arc::new(InMemoryStorage::new());

        // Create multiple experiments
        let spec1 = create_experiment_spec(
            "exp1".to_string(),
            "First experiment".to_string(),
            vec!["tag1".to_string()],
            HashMap::new(),
        );

        let spec2 = create_experiment_spec(
            "exp2".to_string(),
            "Second experiment".to_string(),
            vec!["tag2".to_string()],
            HashMap::new(),
        );

        let mut tracker1 = ExperimentTracker::new(spec1, storage.clone());
        let mut tracker2 = ExperimentTracker::new(spec2, storage.clone());

        tracker1.start().await.unwrap();
        tracker1.complete().await.unwrap();
        tracker2.start().await.unwrap();
        tracker2.complete().await.unwrap();

        // Test filtering
        let filter = ExperimentFilter {
            tags: Some(vec!["tag1".to_string()]),
            ..Default::default()
        };

        let results = storage.list_experiments(Some(&filter)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].spec.name, "exp1");
    }

    #[test]
    fn test_environment_info_collection() {
        let env = collect_environment_info();
        assert!(!env.os.is_empty());
        assert!(!env.rust_version.is_empty());
    }
}
