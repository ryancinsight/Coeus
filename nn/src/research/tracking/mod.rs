//! Advanced Experiment Tracking System
//!
//! This module provides comprehensive experiment metadata tracking,
//! hyperparameter versioning, model checkpoints, and artifact management
//! for research-grade experimentation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// Re-export core types
pub mod artifacts;
pub mod checkpoints;
pub mod hyperparameters;
pub mod metadata;

pub use artifacts::*;
pub use checkpoints::*;
pub use hyperparameters::*;
pub use metadata::*;

/// Enhanced experiment tracker with comprehensive metadata
#[derive(Debug, Clone)]
pub struct ExperimentTracker {
    /// Experiment identifier
    pub experiment_id: String,
    /// Experiment metadata
    pub metadata: ExperimentMetadata,
    /// Hyperparameter tracker
    pub hyperparameters: HyperparameterTracker,
    /// Checkpoint manager
    pub checkpoints: CheckpointManager,
    /// Artifact storage
    pub artifacts: ArtifactStorage,
    /// Experiment tags
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Tracking start time
    pub start_time: Instant,
    /// Version identifier
    pub version: String,
}

impl ExperimentTracker {
    /// Create new experiment tracker
    pub fn new(experiment_id: String, name: String, description: String) -> Self {
        Self {
            experiment_id: experiment_id.clone(),
            metadata: ExperimentMetadata::new(experiment_id, name, description),
            hyperparameters: HyperparameterTracker::new(),
            checkpoints: CheckpointManager::new(),
            artifacts: ArtifactStorage::new(),
            tags: Vec::new(),
            properties: HashMap::new(),
            start_time: Instant::now(),
            version: "1.0.0".to_string(),
        }
    }

    /// Set experiment version
    pub fn with_version(mut self, version: String) -> Self {
        self.version = version;
        self
    }

    /// Add experiment tag
    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Add custom property
    pub fn add_property(mut self, key: String, value: serde_json::Value) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Log hyperparameter with automatic versioning
    pub fn log_hyperparameter(
        &mut self,
        key: String,
        value: serde_json::Value,
        description: Option<String>,
    ) -> crate::core::error::Result<()> {
        self.hyperparameters
            .log_hyperparameter(key, value, description)
    }

    /// Create checkpoint
    pub fn create_checkpoint(
        &mut self,
        name: String,
        data: CheckpointData,
    ) -> crate::core::error::Result<String> {
        self.checkpoints.create_checkpoint(name, data)
    }

    /// Store artifact
    pub fn store_artifact(
        &mut self,
        name: String,
        artifact_type: ArtifactType,
        data: Vec<u8>,
    ) -> crate::core::error::Result<String> {
        self.artifacts.store_artifact(name, artifact_type, data)
    }

    /// Get experiment summary
    pub fn summary(&self) -> ExperimentSummary {
        ExperimentSummary {
            experiment_id: self.experiment_id.clone(),
            metadata: self.metadata.clone(),
            hyperparameter_count: self.hyperparameters.parameters.len(),
            checkpoint_count: self.checkpoints.checkpoints.len(),
            artifact_count: self.artifacts.artifacts.len(),
            tags: self.tags.clone(),
            version: self.version.clone(),
            duration: self.start_time.elapsed(),
            properties: self.properties.clone(),
        }
    }

    /// Export experiment data to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "hyperparameters": self.hyperparameters,
            "checkpoints": self.checkpoints,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "properties": self.properties,
            "version": self.version,
            "duration_seconds": self.start_time.elapsed().as_secs_f64()
        })
    }
}

/// Experiment summary for quick overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub experiment_id: String,
    pub metadata: ExperimentMetadata,
    pub hyperparameter_count: usize,
    pub checkpoint_count: usize,
    pub artifact_count: usize,
    pub tags: Vec<String>,
    pub version: String,
    pub duration: std::time::Duration,
    pub properties: HashMap<String, serde_json::Value>,
}

impl std::fmt::Display for ExperimentSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Experiment {} (v{}): {} HPs, {} CPs, {} artifacts, {:.1}s elapsed",
            self.experiment_id,
            self.version,
            self.hyperparameter_count,
            self.checkpoint_count,
            self.artifact_count,
            self.duration.as_secs_f64()
        )
    }
}

/// Central experiment tracking registry
#[derive(Debug)]
pub struct ExperimentRegistry {
    /// Active experiment trackers
    experiments: Arc<RwLock<HashMap<String, ExperimentTracker>>>,
    /// Archive of completed experiments
    archive: Arc<RwLock<HashMap<String, ExperimentSummary>>>,
    /// Default experiment configurations
    defaults: HashMap<String, ExperimentConfig>,
}

impl ExperimentRegistry {
    /// Create new experiment registry
    pub fn new() -> Self {
        Self {
            experiments: Arc::new(RwLock::new(HashMap::new())),
            archive: Arc::new(RwLock::new(HashMap::new())),
            defaults: HashMap::new(),
        }
    }

    /// Start new experiment
    pub fn start_experiment(
        &self,
        experiment_id: String,
        name: String,
        description: String,
    ) -> ExperimentTracker {
        let tracker = ExperimentTracker::new(experiment_id.clone(), name, description);
        self.experiments
            .write()
            .unwrap()
            .insert(experiment_id, tracker.clone());
        tracker
    }

    /// Get active experiment tracker
    pub fn get_experiment(&self, experiment_id: &str) -> Option<ExperimentTracker> {
        self.experiments.read().unwrap().get(experiment_id).cloned()
    }

    /// Complete experiment and archive it
    pub fn complete_experiment(&self, experiment_id: String) -> crate::core::error::Result<()> {
        let tracker = self.experiments.write().unwrap().remove(&experiment_id);
        if let Some(tracker) = tracker {
            let summary = tracker.summary();
            self.archive.write().unwrap().insert(experiment_id, summary);
        }
        Ok(())
    }

    /// Get archived experiment summary
    pub fn get_archived_experiment(&self, experiment_id: &str) -> Option<ExperimentSummary> {
        self.archive.read().unwrap().get(experiment_id).cloned()
    }

    /// List all active experiments
    pub fn list_active_experiments(&self) -> Vec<String> {
        self.experiments.read().unwrap().keys().cloned().collect()
    }

    /// List all archived experiments
    pub fn list_archived_experiments(&self) -> Vec<String> {
        self.archive.read().unwrap().keys().cloned().collect()
    }

    /// Set default configuration for experiment type
    pub fn set_default_config(&mut self, experiment_type: String, config: ExperimentConfig) {
        self.defaults.insert(experiment_type, config);
    }

    /// Get default configuration for experiment type
    pub fn get_default_config(&self, experiment_type: &str) -> Option<&ExperimentConfig> {
        self.defaults.get(experiment_type)
    }
}

impl Default for ExperimentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Default configuration for experiment types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentConfig {
    /// Default hyperparameters
    pub default_hyperparameters: HashMap<String, serde_json::Value>,
    /// Auto-logging settings
    pub auto_logging: AutoLoggingConfig,
    /// Checkpoint policy
    pub checkpoint_policy: CheckpointPolicy,
    /// Artifact retention policy
    pub artifact_retention: ArtifactRetentionPolicy,
}

/// Auto-logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLoggingConfig {
    /// Log system metrics (CPU, memory, etc.)
    pub log_system_metrics: bool,
    /// Log GPU metrics
    pub log_gpu_metrics: bool,
    /// Log interval in seconds
    pub log_interval_seconds: u64,
    /// Maximum log entries
    pub max_log_entries: usize,
}

impl Default for AutoLoggingConfig {
    fn default() -> Self {
        Self {
            log_system_metrics: true,
            log_gpu_metrics: true,
            log_interval_seconds: 30,
            max_log_entries: 10000,
        }
    }
}

/// Checkpoint policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPolicy {
    /// Auto-save interval in seconds
    pub auto_save_interval_seconds: Option<u64>,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Checkpoint only on improvement
    pub checkpoint_on_improvement: bool,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            auto_save_interval_seconds: Some(300), // 5 minutes
            max_checkpoints: 10,
            checkpoint_on_improvement: true,
        }
    }
}

/// Artifact retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRetentionPolicy {
    /// Maximum artifact size in MB
    pub max_artifact_size_mb: usize,
    /// Retention period in days
    pub retention_days: usize,
    /// Auto-compress large artifacts
    pub auto_compress: bool,
}

impl Default for ArtifactRetentionPolicy {
    fn default() -> Self {
        Self {
            max_artifact_size_mb: 100,
            retention_days: 365,
            auto_compress: true,
        }
    }
}
