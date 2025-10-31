//! CLIP Experiment Tracking
//!
//! Specialized tracking for CLIP experiments including
//! hyperparameters, metrics, artifacts, and reproducibility data.

use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// CLIP experiment tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipExperimentTracking {
    /// Experiment ID
    pub experiment_id: String,
    /// CLIP-specific hyperparameters
    pub clip_hyperparameters: ClipHyperparameters,
    /// Training metrics over time
    pub training_metrics: Vec<TrainingMetricsEntry>,
    /// Validation metrics over time
    pub validation_metrics: Vec<ValidationMetricsEntry>,
    /// CLIP-specific metrics
    pub clip_metrics: ClipMetrics,
    /// Experiment artifacts
    pub artifacts: Vec<ExperimentArtifact>,
    /// Reproducibility information
    pub reproducibility: ReproducibilityInfo,
}

/// CLIP-specific hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipHyperparameters {
    /// Vision encoder parameters
    pub vision_params: VisionParameters,
    /// Text encoder parameters
    pub text_params: TextParameters,
    /// Training parameters
    pub training_params: ClipTrainingParameters,
}

/// Vision encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionParameters {
    pub architecture: String,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub image_size: usize,
}

/// Text encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextParameters {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub max_seq_length: usize,
    pub context_length: usize,
}

/// CLIP training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipTrainingParameters {
    pub temperature: f64,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub max_grad_norm: f64,
    pub num_epochs: usize,
}

/// Training metrics entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetricsEntry {
    pub epoch: usize,
    pub step: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Validation metrics entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetricsEntry {
    pub epoch: usize,
    pub loss: f64,
    pub clip_metrics: ClipMetrics,
    pub timestamp: DateTime<Utc>,
}

/// CLIP-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipMetrics {
    /// Image-to-text retrieval metrics
    pub i2t_retrieval: RetrievalMetrics,
    /// Text-to-image retrieval metrics
    pub t2i_retrieval: RetrievalMetrics,
    /// Zero-shot classification accuracy
    pub zero_shot_accuracy: f64,
    /// CLIP loss components
    pub clip_loss: ClipLossComponents,
}

/// Retrieval metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub r1: f64,  // Recall@1
    pub r5: f64,  // Recall@5
    pub r10: f64, // Recall@10
    pub median_rank: f64,
    pub mean_rank: f64,
}

/// CLIP loss components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipLossComponents {
    pub image_loss: f64,
    pub text_loss: f64,
    pub total_loss: f64,
    pub temperature: f64,
}

/// Experiment artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentArtifact {
    pub name: String,
    pub artifact_type: ArtifactType,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub description: String,
}

/// Artifact types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    ModelCheckpoint,
    TrainingLog,
    MetricsPlot,
    ConfigFile,
    DatasetSample,
    EvaluationResults,
}

/// Reproducibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    pub random_seed: u64,
    pub framework_version: String,
    pub cuda_version: Option<String>,
    pub gpu_model: Option<String>,
    pub cpu_model: String,
    pub ram_gb: usize,
    pub environment_variables: HashMap<String, String>,
    pub git_commit: Option<String>,
    pub command_line: Vec<String>,
}

/// CLIP experiment tracker
pub struct ClipExperimentTracker {
    experiment_id: String,
    output_dir: PathBuf,
    tracking_data: ClipExperimentTracking,
}

impl ClipExperimentTracker {
    /// Create new CLIP experiment tracker
    pub fn new(experiment_id: String, output_dir: PathBuf) -> Self {
        let tracking_data = ClipExperimentTracking {
            experiment_id: experiment_id.clone(),
            clip_hyperparameters: ClipHyperparameters {
                vision_params: VisionParameters {
                    architecture: "ViT-B/16".to_string(),
                    patch_size: 16,
                    hidden_dim: 768,
                    num_layers: 12,
                    num_heads: 12,
                    image_size: 224,
                },
                text_params: TextParameters {
                    architecture: "Transformer".to_string(),
                    vocab_size: 49408,
                    hidden_dim: 512,
                    num_layers: 12,
                    max_seq_length: 77,
                    context_length: 77,
                },
                training_params: ClipTrainingParameters {
                    temperature: 0.07,
                    batch_size: 32,
                    learning_rate: 5e-4,
                    weight_decay: 0.2,
                    warmup_steps: 1000,
                    max_grad_norm: 1.0,
                    num_epochs: 10,
                },
            },
            training_metrics: Vec::new(),
            validation_metrics: Vec::new(),
            clip_metrics: ClipMetrics {
                i2t_retrieval: RetrievalMetrics {
                    r1: 0.0,
                    r5: 0.0,
                    r10: 0.0,
                    median_rank: 0.0,
                    mean_rank: 0.0,
                },
                t2i_retrieval: RetrievalMetrics {
                    r1: 0.0,
                    r5: 0.0,
                    r10: 0.0,
                    median_rank: 0.0,
                    mean_rank: 0.0,
                },
                zero_shot_accuracy: 0.0,
                clip_loss: ClipLossComponents {
                    image_loss: 0.0,
                    text_loss: 0.0,
                    total_loss: 0.0,
                    temperature: 0.07,
                },
            },
            artifacts: Vec::new(),
            reproducibility: ReproducibilityInfo {
                random_seed: 42,
                framework_version: env!("CARGO_PKG_VERSION").to_string(),
                cuda_version: None,
                gpu_model: None,
                cpu_model: "Unknown".to_string(),
                ram_gb: 0,
                environment_variables: std::env::vars().collect(),
                git_commit: None,
                command_line: std::env::args().collect(),
            },
        };

        Self {
            experiment_id,
            output_dir,
            tracking_data,
        }
    }

    /// Log training metrics
    pub fn log_training_metrics(&mut self, epoch: usize, step: usize, loss: f64, lr: f64, grad_norm: Option<f64>) {
        let entry = TrainingMetricsEntry {
            epoch,
            step,
            loss,
            learning_rate: lr,
            grad_norm,
            timestamp: Utc::now(),
        };
        self.tracking_data.training_metrics.push(entry);
    }

    /// Log validation metrics
    pub fn log_validation_metrics(&mut self, epoch: usize, loss: f64, clip_metrics: ClipMetrics) {
        let entry = ValidationMetricsEntry {
            epoch,
            loss,
            clip_metrics,
            timestamp: Utc::now(),
        };
        self.tracking_data.validation_metrics.push(entry);
    }

    /// Update CLIP metrics
    pub fn update_clip_metrics(&mut self, metrics: ClipMetrics) {
        self.tracking_data.clip_metrics = metrics;
    }

    /// Add artifact
    pub fn add_artifact(&mut self, name: String, artifact_type: ArtifactType, path: PathBuf, description: String) {
        let size_bytes = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(0);

        let artifact = ExperimentArtifact {
            name,
            artifact_type,
            path,
            size_bytes,
            created_at: Utc::now(),
            description,
        };

        self.tracking_data.artifacts.push(artifact);
    }

    /// Save tracking data to file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let tracking_file = self.output_dir.join(format!("{}_tracking.json", self.experiment_id));
        let json = serde_json::to_string_pretty(&self.tracking_data)?;
        std::fs::write(tracking_file, json)?;
        Ok(())
    }

    /// Get current metrics summary
    pub fn get_metrics_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        // Add latest training metrics
        if let Some(latest) = self.tracking_data.training_metrics.last() {
            summary.insert("latest_training_loss".to_string(), latest.loss);
            summary.insert("latest_learning_rate".to_string(), latest.learning_rate);
        }

        // Add latest validation metrics
        if let Some(latest) = self.tracking_data.validation_metrics.last() {
            summary.insert("latest_validation_loss".to_string(), latest.loss);
            summary.insert("r1_i2t".to_string(), latest.clip_metrics.i2t_retrieval.r1);
            summary.insert("r5_i2t".to_string(), latest.clip_metrics.i2t_retrieval.r5);
            summary.insert("r10_i2t".to_string(), latest.clip_metrics.i2t_retrieval.r10);
            summary.insert("r1_t2i".to_string(), latest.clip_metrics.t2i_retrieval.r1);
            summary.insert("r5_t2i".to_string(), latest.clip_metrics.t2i_retrieval.r5);
            summary.insert("r10_t2i".to_string(), latest.clip_metrics.t2i_retrieval.r10);
            summary.insert("zero_shot_accuracy".to_string(), latest.clip_metrics.zero_shot_accuracy);
        }

        summary
    }

    /// Get tracking data reference
    pub fn tracking_data(&self) -> &ClipExperimentTracking {
        &self.tracking_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_experiment_tracker_creation() {
        let tracker = ClipExperimentTracker::new(
            "test_experiment".to_string(),
            PathBuf::from("./test_output"),
        );

        assert_eq!(tracker.experiment_id, "test_experiment");
        assert_eq!(tracker.tracking_data.clip_hyperparameters.training_params.temperature, 0.07);
    }

    #[test]
    fn test_log_training_metrics() {
        let mut tracker = ClipExperimentTracker::new(
            "test".to_string(),
            PathBuf::from("./test"),
        );

        tracker.log_training_metrics(1, 100, 2.5, 0.001, Some(1.2));

        assert_eq!(tracker.tracking_data.training_metrics.len(), 1);
        let metrics = &tracker.tracking_data.training_metrics[0];
        assert_eq!(metrics.epoch, 1);
        assert_eq!(metrics.step, 100);
        assert_eq!(metrics.loss, 2.5);
        assert_eq!(metrics.learning_rate, 0.001);
        assert_eq!(metrics.grad_norm, Some(1.2));
    }

    #[test]
    fn test_metrics_summary() {
        let mut tracker = ClipExperimentTracker::new(
            "test".to_string(),
            PathBuf::from("./test"),
        );

        // Add some metrics
        tracker.log_training_metrics(1, 100, 2.5, 0.001, None);

        let clip_metrics = ClipMetrics {
            i2t_retrieval: RetrievalMetrics {
                r1: 0.5,
                r5: 0.8,
                r10: 0.9,
                median_rank: 2.0,
                mean_rank: 3.5,
            },
            t2i_retrieval: RetrievalMetrics {
                r1: 0.4,
                r5: 0.7,
                r10: 0.85,
                median_rank: 3.0,
                mean_rank: 4.2,
            },
            zero_shot_accuracy: 0.65,
            clip_loss: ClipLossComponents {
                image_loss: 1.2,
                text_loss: 1.3,
                total_loss: 2.5,
                temperature: 0.07,
            },
        };

        tracker.log_validation_metrics(1, 2.5, clip_metrics);

        let summary = tracker.get_metrics_summary();

        assert_eq!(summary.get("latest_training_loss"), Some(&2.5));
        assert_eq!(summary.get("latest_validation_loss"), Some(&2.5));
        assert_eq!(summary.get("r1_i2t"), Some(&0.5));
        assert_eq!(summary.get("zero_shot_accuracy"), Some(&0.65));
    }

    #[test]
    fn test_retrieval_metrics_structure() {
        let metrics = RetrievalMetrics {
            r1: 0.5,
            r5: 0.8,
            r10: 0.9,
            median_rank: 2.0,
            mean_rank: 3.5,
        };

        assert_eq!(metrics.r1, 0.5);
        assert_eq!(metrics.r5, 0.8);
        assert_eq!(metrics.r10, 0.9);
        assert_eq!(metrics.median_rank, 2.0);
        assert_eq!(metrics.mean_rank, 3.5);
    }
}





