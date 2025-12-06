//! CLIP Experiment Builder
//!
//! Provides a fluent API for constructing CLIP experiments with
//! systematic hyperparameter optimization, ablation studies, and
//! automated experiment management.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{NNError, Result};
use crate::clip::enhanced_trainer::{EnhancedClipTrainingConfig, EnhancedClipTrainer};
use super::{ClipResearchConfig, HpoSpace, AblationStudy, ClipExperimentMetadata};

/// CLIP experiment builder for systematic research
#[derive(Debug, Clone)]
pub struct ClipExperimentBuilder {
    config: ClipResearchConfig,
    experiment_name: String,
    output_dir: PathBuf,
    tags: HashMap<String, String>,
    priority: ExperimentPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExperimentPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for ExperimentPriority {
    fn default() -> Self {
        ExperimentPriority::Normal
    }
}

impl ClipExperimentBuilder {
    /// Create new experiment builder
    pub fn new(experiment_name: impl Into<String>) -> Self {
        Self {
            config: ClipResearchConfig::default(),
            experiment_name: experiment_name.into(),
            output_dir: PathBuf::from("./clip_experiments"),
            tags: HashMap::new(),
            priority: ExperimentPriority::default(),
        }
    }

    /// Set base training configuration
    pub fn with_training_config(mut self, config: EnhancedClipTrainingConfig) -> Self {
        self.config.base_training_config = config;
        self
    }

    /// Add HPO search space
    pub fn with_hpo_space(mut self, space: HpoSpace) -> Self {
        self.config.hpo_spaces.push(space);
        self
    }

    /// Add ablation study
    pub fn with_ablation_study(mut self, study: AblationStudy) -> Self {
        self.config.ablation_configs.push(study);
        self
    }

    /// Set output directory
    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = dir;
        self
    }

    /// Add experiment tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set experiment priority
    pub fn with_priority(mut self, priority: ExperimentPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Build comprehensive CLIP experiment
    pub fn build(self) -> Result<ClipExperiment> {
        // Validate configuration
        self.validate_config()?;

        // Create experiment metadata
        let mut metadata = ClipExperimentMetadata::default();
        metadata.name = self.experiment_name.clone();
        metadata.description = format!("CLIP experiment: {}", self.experiment_name);
        metadata.author = std::env::var("USER").unwrap_or_else(|_| "coeus".to_string());

        // Convert tags HashMap to Vec<String> for tags field
        metadata.tags = self.tags.keys().cloned().collect();

        // Update config with metadata
        let mut config = self.config;
        config.metadata = metadata;

        Ok(ClipExperiment {
            name: self.experiment_name,
            config,
            output_dir: self.output_dir,
            priority: self.priority,
            status: ExperimentStatus::Pending,
        })
    }

    /// Validate experiment configuration
    fn validate_config(&self) -> Result<()> {
        // Validate base training config
        if self.config.base_training_config.base_config.num_epochs == 0 {
            return Err(NNError::InvalidInput {
                message: "Number of epochs must be greater than 0".to_string(),
            });
        }

        // Validate HPO spaces
        for space in &self.config.hpo_spaces {
            if space.dimensions.is_empty() {
                return Err(NNError::InvalidInput {
                    message: format!("HPO space '{}' has no dimensions", space.name),
                });
            }
        }

        // Validate output directory
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir).map_err(|e| NNError::IoError {
                error: e,
            })?;
        }

        Ok(())
    }
}

/// Built CLIP experiment
#[derive(Debug)]
pub struct ClipExperiment {
    pub name: String,
    pub config: ClipResearchConfig,
    pub output_dir: PathBuf,
    pub priority: ExperimentPriority,
    pub status: ExperimentStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// CLIP experiment runner
pub struct ClipExperimentRunner {
    experiment: ClipExperiment,
}

impl ClipExperimentRunner {
    /// Create experiment runner
    pub fn new(experiment: ClipExperiment) -> Self {
        Self { experiment }
    }

    /// Run the experiment
    pub async fn run(self) -> Result<ExperimentResult> {
        println!("🚀 Starting CLIP experiment: {}", self.experiment.name);

        // Initialize trainer
        let mut trainer = EnhancedClipTrainer::<backend::CpuBackend<dtype::float::Float32>, storage::DenseStorage<dtype::float::Float32>, dtype::float::Float32>::new(self.experiment.config.base_training_config.clone())
            .map_err(|e| NNError::TrainingError {
                message: format!("Failed to initialize trainer: {}", e),
            })?;

        // Run training
        let data_loader: fn() -> Option<crate::clip::enhanced_trainer::ClipBatch> = || None; // Placeholder - no data
        let training_result = trainer.train(data_loader).await
            .map_err(|e| NNError::TrainingError {
                message: format!("Training failed: {}", e),
            })?;

        println!("✅ CLIP experiment completed: {}", self.experiment.name);

        Ok(ExperimentResult {
            experiment_name: self.experiment.name,
            status: ExperimentStatus::Completed,
            training_result,
            metrics: HashMap::new(), // TODO: Add comprehensive metrics
            artifacts: Vec::new(),   // TODO: Add experiment artifacts
            execution_time: std::time::Duration::from_secs(0), // TODO: Track actual time
        })
    }
}

/// Experiment execution result
#[derive(Debug)]
pub struct ExperimentResult {
    pub experiment_name: String,
    pub status: ExperimentStatus,
    pub training_result: crate::clip::enhanced_trainer::EnhancedTrainingReport,
    pub metrics: HashMap<String, f64>,
    pub artifacts: Vec<String>,
    pub execution_time: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_builder_creation() {
        let builder = ClipExperimentBuilder::new("test_experiment");
        assert_eq!(builder.experiment_name, "test_experiment");
        assert_eq!(builder.priority, ExperimentPriority::Normal);
    }

    #[test]
    fn test_experiment_builder_with_tag() {
        let builder = ClipExperimentBuilder::new("test")
            .with_tag("dataset", "coco")
            .with_tag("model", "clip-vit");

        assert_eq!(builder.tags.get("dataset"), Some(&"coco".to_string()));
        assert_eq!(builder.tags.get("model"), Some(&"clip-vit".to_string()));
    }

    #[test]
    fn test_experiment_builder_with_priority() {
        let builder = ClipExperimentBuilder::new("test")
            .with_priority(ExperimentPriority::High);

        assert_eq!(builder.priority, ExperimentPriority::High);
    }

    #[test]
    fn test_experiment_priority_ordering() {
        assert!(ExperimentPriority::Low < ExperimentPriority::Normal);
        assert!(ExperimentPriority::Normal < ExperimentPriority::High);
        assert!(ExperimentPriority::High < ExperimentPriority::Critical);
    }
}
