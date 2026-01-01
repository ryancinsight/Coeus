//! Model Checkpoint Management System
//!
//! This module provides comprehensive model checkpoint management for research
//! experiments, including automatic saving, versioning, restoration, and
//! research-grade checkpoint analysis capabilities.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Checkpoint data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Checkpoint identifier
    pub id: String,
    /// Checkpoint name/description
    pub name: String,
    /// Model state data (serialized tensors, parameters, etc.)
    pub model_state: HashMap<String, serde_json::Value>,
    /// Optimizer state data
    pub optimizer_state: HashMap<String, serde_json::Value>,
    /// Training state (epoch, step, etc.)
    pub training_state: TrainingState,
    /// Metadata about the checkpoint
    pub metadata: CheckpointMetadata,
    /// Performance metrics at checkpoint time
    pub performance_metrics: HashMap<String, f64>,
    /// Size of checkpoint in bytes
    pub size_bytes: u64,
}

/// Training state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: u64,
    /// Current step within epoch
    pub step: u64,
    /// Total steps completed
    pub total_steps: u64,
    /// Learning rate at checkpoint
    pub learning_rate: f64,
    /// Best validation performance seen
    pub best_performance: Option<f64>,
    /// Training loss
    pub loss: f64,
    /// Validation metrics
    pub validation_metrics: HashMap<String, f64>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// When checkpoint was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Framework version used
    pub framework_version: String,
    /// Hardware information
    pub hardware_info: String,
    /// Checkpoint version
    pub version: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Notes/comments
    pub notes: String,
}

/// Checkpoint manager for experiment lifecycle
#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckpointManager {
    /// All checkpoints, indexed by ID
    pub checkpoints: HashMap<String, CheckpointData>,
    /// Checkpoints sorted by creation time (for easy retrieval)
    checkpoints_by_time: BTreeMap<chrono::DateTime<chrono::Utc>, String>,
    /// Best performing checkpoints
    best_checkpoints: Vec<String>,
    /// Storage configuration
    storage_config: CheckpointStorageConfig,
    /// Auto-save scheduler
    auto_save_scheduler: AutoSaveScheduler,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
            checkpoints_by_time: BTreeMap::new(),
            best_checkpoints: Vec::new(),
            storage_config: CheckpointStorageConfig::default(),
            auto_save_scheduler: AutoSaveScheduler::new(),
        }
    }

    /// Create a checkpoint
    pub fn create_checkpoint(
        &mut self,
        name: String,
        data: CheckpointData,
    ) -> crate::error::Result<String> {
        let checkpoint_id = format!("{}_{}", name, chrono::Utc::now().timestamp());
        let mut data = data;
        data.id = checkpoint_id.clone();

        // Update temporal index
        self.checkpoints_by_time
            .insert(data.metadata.created_at, checkpoint_id.clone());

        // Update best checkpoints list
        self.update_best_checkpoints(&data);

        // Store checkpoint
        self.checkpoints.insert(checkpoint_id.clone(), data);

        // Clean up old checkpoints if needed
        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_id)
    }

    /// Get checkpoint by ID
    pub fn get_checkpoint(&self, id: &str) -> Option<&CheckpointData> {
        self.checkpoints.get(id)
    }

    /// Get latest checkpoint
    pub fn get_latest_checkpoint(&self) -> Option<&CheckpointData> {
        self.checkpoints_by_time
            .values()
            .last()
            .and_then(|id| self.checkpoints.get(id))
    }

    /// Get best performing checkpoint
    pub fn get_best_checkpoint(&self, metric: Option<&str>) -> Option<&CheckpointData> {
        if let Some(metric) = metric {
            // Find checkpoint with best value for specific metric
            self.checkpoints
                .values()
                .filter(|cp| cp.performance_metrics.contains_key(metric))
                .max_by(|a, b| {
                    let a_val = a.performance_metrics[metric];
                    let b_val = b.performance_metrics[metric];
                    a_val.partial_cmp(&b_val).unwrap()
                })
        } else if !self.best_checkpoints.is_empty() {
            // Return checkpoint with best overall performance
            self.best_checkpoints
                .first()
                .and_then(|id| self.checkpoints.get(id))
        } else {
            None
        }
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Vec<&CheckpointData> {
        self.checkpoints.values().collect()
    }

    /// List checkpoints with filtering
    pub fn list_checkpoints_filtered<F>(&self, filter: F) -> Vec<&CheckpointData>
    where
        F: Fn(&&CheckpointData) -> bool,
    {
        self.checkpoints.values().filter(filter).collect()
    }

    /// Delete checkpoint
    pub fn delete_checkpoint(&mut self, id: &str) -> crate::error::Result<()> {
        if let Some(checkpoint) = self.checkpoints.remove(id) {
            self.checkpoints_by_time
                .remove(&checkpoint.metadata.created_at);
            self.best_checkpoints.retain(|best_id| best_id != id);
        }
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load_checkpoint(&mut self, path: &Path) -> crate::error::Result<String> {
        // Implementation would depend on storage format
        // For now, return error indicating not implemented
        Err(crate::error::NNError::NotImplemented {
            operation: "File-based checkpoint loading not yet implemented".to_string(),
        })
    }

    /// Save checkpoint to file
    pub fn save_checkpoint(&self, id: &str, path: &Path) -> crate::error::Result<()> {
        // Implementation would serialize checkpoint to file
        let _ = (id, path); // Suppress unused variable warnings
        Err(crate::error::NNError::NotImplemented {
            operation: "File-based checkpoint saving not yet implemented".to_string(),
        })
    }

    /// Generate checkpoint comparison report
    pub fn comparison_report(&self, checkpoint_ids: &[String]) -> CheckpointComparisonReport {
        let mut checkpoints = Vec::new();

        for id in checkpoint_ids {
            if let Some(cp) = self.checkpoints.get(id) {
                checkpoints.push(cp.clone());
            }
        }

        CheckpointComparisonReport {
            checkpoints,
            metric_improvements: self.calculate_metric_improvements(checkpoint_ids),
            training_efficiency: self.calculate_training_efficiency(checkpoint_ids),
        }
    }

    /// Schedule auto-save checkpoint
    pub fn schedule_auto_save(&mut self, interval_seconds: u64, metric: Option<String>) {
        self.auto_save_scheduler.schedule(interval_seconds, metric);
    }

    /// Check if auto-save should trigger
    pub fn should_auto_save(&self, current_metrics: &HashMap<String, f64>) -> bool {
        self.auto_save_scheduler.should_save(current_metrics)
    }

    /// Update best checkpoints list
    fn update_best_checkpoints(&mut self, new_checkpoint: &CheckpointData) {
        // Simple strategy: keep top 3 performing checkpoints
        self.best_checkpoints.push(new_checkpoint.id.clone());
        self.best_checkpoints.sort_by(|a, b| {
            let a_score = self
                .checkpoints
                .get(a)
                .and_then(|cp| {
                    cp.performance_metrics
                        .get("validation_accuracy")
                        .or_else(|| cp.performance_metrics.get("loss"))
                })
                .unwrap_or(&f64::INFINITY);
            let b_score = self
                .checkpoints
                .get(b)
                .and_then(|cp| {
                    cp.performance_metrics
                        .get("validation_accuracy")
                        .or_else(|| cp.performance_metrics.get("loss"))
                })
                .unwrap_or(&f64::INFINITY);
            // For accuracy: higher is better, for loss: lower is better
            if new_checkpoint
                .performance_metrics
                .contains_key("validation_accuracy")
            {
                b_score.partial_cmp(a_score).unwrap()
            } else {
                a_score.partial_cmp(b_score).unwrap()
            }
        });
        self.best_checkpoints.truncate(3);
    }

    /// Clean up old checkpoints based on policy
    fn cleanup_old_checkpoints(&mut self) -> crate::error::Result<()> {
        if self.checkpoints.len() <= self.storage_config.max_checkpoints {
            return Ok(());
        }

        // Remove oldest checkpoints beyond the limit
        let to_remove: Vec<String> = self
            .checkpoints_by_time
            .values()
            .take(self.checkpoints.len() - self.storage_config.max_checkpoints)
            .cloned()
            .collect();

        for id in to_remove {
            self.delete_checkpoint(&id)?;
        }

        Ok(())
    }

    /// Calculate metric improvements between checkpoints
    fn calculate_metric_improvements(
        &self,
        checkpoint_ids: &[String],
    ) -> HashMap<String, Vec<f64>> {
        let mut improvements = HashMap::new();
        let mut prev_metrics = HashMap::new();

        for id in checkpoint_ids {
            if let Some(cp) = self.checkpoints.get(id) {
                for (metric_name, current_value) in &cp.performance_metrics {
                    let prev_value = prev_metrics
                        .get(metric_name)
                        .copied()
                        .unwrap_or(*current_value);
                    let improvement = current_value - prev_value;
                    improvements
                        .entry(metric_name.clone())
                        .or_insert_with(Vec::new)
                        .push(improvement);
                    prev_metrics.insert(metric_name.clone(), *current_value);
                }
            }
        }

        improvements
    }

    /// Calculate training efficiency metrics
    fn calculate_training_efficiency(
        &self,
        checkpoint_ids: &[String],
    ) -> TrainingEfficiencyMetrics {
        let mut total_steps = 0u64;
        let mut total_time = std::time::Duration::ZERO;
        let mut first_time = None;

        for id in checkpoint_ids {
            if let Some(cp) = self.checkpoints.get(id) {
                total_steps += cp.training_state.total_steps;
                if first_time.is_none() {
                    first_time = Some(cp.metadata.created_at);
                }
                // Calculate time span
                if let Some(first) = first_time {
                    let span = cp.metadata.created_at.signed_duration_since(first);
                    total_time = total_time.max(span.to_std().unwrap_or(std::time::Duration::ZERO));
                }
            }
        }

        TrainingEfficiencyMetrics {
            checkpoints_per_second: if total_time.as_secs() > 0 {
                checkpoint_ids.len() as f64 / total_time.as_secs() as f64
            } else {
                0.0
            },
            steps_per_checkpoint: if checkpoint_ids.is_empty() {
                0.0
            } else {
                total_steps as f64 / checkpoint_ids.len() as f64
            },
            average_checkpoint_interval: if checkpoint_ids.len() > 1 && total_time.as_secs() > 0 {
                total_time.as_secs() as f64 / (checkpoint_ids.len() - 1) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointStorageConfig {
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Maximum storage size (bytes)
    pub max_storage_size: u64,
    /// Whether to compress checkpoints
    pub compress: bool,
    /// Storage directory
    pub storage_dir: PathBuf,
    /// File format for checkpoints
    pub format: CheckpointFormat,
}

impl Default for CheckpointStorageConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            max_storage_size: 10 * 1024 * 1024 * 1024, // 10GB
            compress: true,
            storage_dir: PathBuf::from("./checkpoints"),
            format: CheckpointFormat::Safetensors,
        }
    }
}

/// Checkpoint file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// SafeTensors format (recommended)
    Safetensors,
    /// Pickle format (Python compatibility)
    Pickle,
    /// JSON format (human readable)
    Json,
    /// Custom binary format
    Binary,
}

/// Auto-save scheduler for automatic checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSaveScheduler {
    /// Save interval in seconds
    pub interval_seconds: Option<u64>,
    /// Metric to monitor for best checkpoints
    pub monitor_metric: Option<String>,
    /// Whether higher values are better for the metric
    pub higher_is_better: bool,
    /// Current best value seen
    pub current_best: Option<f64>,
    /// Last save time
    pub last_save_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl AutoSaveScheduler {
    /// Create new auto-save scheduler
    pub fn new() -> Self {
        Self {
            interval_seconds: None,
            monitor_metric: None,
            higher_is_better: true,
            current_best: None,
            last_save_time: None,
        }
    }

    /// Schedule automatic saves
    pub fn schedule(&mut self, interval_seconds: u64, monitor_metric: Option<String>) {
        self.interval_seconds = Some(interval_seconds);
        self.higher_is_better = monitor_metric
            .as_ref()
            .map(|m| !m.contains("loss"))
            .unwrap_or(true);
        self.monitor_metric = monitor_metric;
    }

    /// Check if save should be triggered
    pub fn should_save(&self, current_metrics: &HashMap<String, f64>) -> bool {
        // Check time-based saving
        if let Some(interval) = self.interval_seconds {
            if let Some(last_time) = self.last_save_time {
                let elapsed = chrono::Utc::now()
                    .signed_duration_since(last_time)
                    .num_seconds();
                if elapsed >= interval as i64 {
                    return true;
                }
            } else {
                // First save
                return true;
            }
        }

        // Check metric-based saving
        if let Some(metric) = &self.monitor_metric {
            if let Some(current_value) = current_metrics.get(metric) {
                match (self.current_best, self.higher_is_better) {
                    (Some(best), true) if *current_value > best => return true,
                    (Some(best), false) if *current_value < best => return true,
                    (None, _) => return true,
                    _ => {}
                }
            }
        }

        false
    }

    /// Update last save time
    pub fn update_last_save(&mut self, best_value: Option<f64>) {
        self.last_save_time = Some(chrono::Utc::now());
        if let Some(value) = best_value {
            self.current_best = Some(value);
        }
    }
}

impl Default for AutoSaveScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointComparisonReport {
    /// Checkpoints being compared
    pub checkpoints: Vec<CheckpointData>,
    /// Metric improvements between checkpoints
    pub metric_improvements: HashMap<String, Vec<f64>>,
    /// Training efficiency metrics
    pub training_efficiency: TrainingEfficiencyMetrics,
}

/// Training efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEfficiencyMetrics {
    /// Checkpoints created per second
    pub checkpoints_per_second: f64,
    /// Average steps between checkpoints
    pub steps_per_checkpoint: f64,
    /// Average time interval between checkpoints (seconds)
    pub average_checkpoint_interval: f64,
}

impl Default for CheckpointData {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "default_checkpoint".to_string(),
            model_state: HashMap::new(),
            optimizer_state: HashMap::new(),
            training_state: TrainingState::default(),
            metadata: CheckpointMetadata::default(),
            performance_metrics: HashMap::new(),
            size_bytes: 0,
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_steps: 0,
            learning_rate: 0.001,
            best_performance: None,
            loss: 0.0,
            validation_metrics: HashMap::new(),
            random_state: None,
        }
    }
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            hardware_info: "Unknown".to_string(),
            version: "1.0".to_string(),
            tags: Vec::new(),
            notes: String::new(),
        }
    }
}
