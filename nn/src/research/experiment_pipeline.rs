//! Experimental pipeline for automated research workflows
//!
//! This module provides a high-level interface for running experimental pipelines
//! that combine multiple research techniques like NAS, HPO, and meta-learning.

use crate::error::{NNError, Result};
use crate::research::tracking::{ExperimentSummary, ExperimentTracker};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// Configuration for an experimental pipeline
#[derive(Debug, Clone)]
pub struct ExperimentPipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Maximum number of experiments to run
    pub max_experiments: usize,
    /// Parallel execution limit
    pub parallel_limit: usize,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (number of experiments without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
}

/// Result of running an experimental pipeline
#[derive(Debug, Clone)]
pub struct ExperimentPipelineResult {
    /// Pipeline configuration used
    pub config: ExperimentPipelineConfig,
    /// All experiment summaries
    pub experiments: Vec<ExperimentSummary>,
    /// Best experiment found
    pub best_experiment: Option<ExperimentSummary>,
    /// Pipeline execution statistics
    pub statistics: PipelineStatistics,
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    /// Total execution time
    pub total_time: std::time::Duration,
    /// Number of experiments completed
    pub experiments_completed: usize,
    /// Number of experiments failed
    pub experiments_failed: usize,
    /// Average experiment time
    pub avg_experiment_time: std::time::Duration,
}

/// Experimental pipeline executor
pub struct ExperimentPipeline {
    config: ExperimentPipelineConfig,
    tracker: Arc<ExperimentTracker>,
}

impl ExperimentPipeline {
    /// Create a new experimental pipeline
    pub fn new(config: ExperimentPipelineConfig, tracker: Arc<ExperimentTracker>) -> Self {
        Self { config, tracker }
    }

    /// Execute the experimental pipeline with parallel execution
    pub async fn execute<F>(&self, experiment_fn: F) -> Result<ExperimentPipelineResult>
    where
        F: Fn() -> Result<ExperimentSummary> + Send + Sync + Clone + 'static,
    {
        let start_time = std::time::Instant::now();

        // Create semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.parallel_limit));
        let mut handles = Vec::new();

        // Launch experiments with controlled parallelism
        for i in 0..self.config.max_experiments {
            let permit =
                semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|e| NNError::ResourceError {
                        message: format!("Failed to acquire execution permit: {}", e),
                    })?;

            let experiment_fn_clone = experiment_fn.clone();

            // Create the async task
            let task = async move {
                let _permit = permit; // Hold permit for duration

                experiment_fn_clone()
            };

            let handle = tokio::spawn(task);
            handles.push(handle);

            // Throttle experiment launches to prevent overwhelming the system
            if i % self.config.parallel_limit == 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }

        // Wait for all experiments to complete and collect results
        let mut experiments = Vec::new();
        let mut best_experiment = None;
        let mut experiments_completed = 0;
        let mut experiments_failed = 0;

        for handle in handles {
            match handle.await {
                Ok(Ok(summary)) => {
                    experiments_completed += 1;

                    // Update best experiment
                    let current_metric = summary
                        .properties
                        .get("accuracy")
                        .and_then(|v| v.as_f64())
                        .or_else(|| {
                            summary
                                .properties
                                .get("loss")
                                .and_then(|v| v.as_f64())
                                .map(|v| -v)
                        })
                        .unwrap_or(0.0);

                    let best_metric = best_experiment
                        .as_ref()
                        .and_then(|b: &ExperimentSummary| {
                            b.properties.get("accuracy").and_then(|v| v.as_f64())
                        })
                        .or_else(|| {
                            best_experiment.as_ref().and_then(|b: &ExperimentSummary| {
                                b.properties
                                    .get("loss")
                                    .and_then(|v| v.as_f64())
                                    .map(|v: f64| -v)
                            })
                        })
                        .unwrap_or(f64::NEG_INFINITY);

                    if best_experiment.is_none() || current_metric > best_metric {
                        best_experiment = Some(summary.clone());
                    }

                    experiments.push(summary);
                }
                Ok(Err(e)) => {
                    experiments_failed += 1;
                    tracing::error!("Experiment failed: {}", e);
                }
                Err(e) => {
                    experiments_failed += 1;
                    tracing::error!("Task join error: {}", e);
                }
            }
        }

        let total_time = start_time.elapsed();

        let avg_experiment_time = if experiments_completed > 0 {
            total_time / experiments_completed as u32
        } else {
            std::time::Duration::from_secs(0)
        };

        Ok(ExperimentPipelineResult {
            config: self.config.clone(),
            experiments,
            best_experiment,
            statistics: PipelineStatistics {
                total_time,
                experiments_completed,
                experiments_failed,
                avg_experiment_time,
            },
        })
    }

    /// Check if early stopping criteria are met
    fn should_early_stop(
        &self,
        experiments: &[ExperimentSummary],
        config: &EarlyStoppingConfig,
    ) -> bool {
        if experiments.len() < config.patience {
            return false;
        }

        let recent_experiments = &experiments[experiments.len().saturating_sub(config.patience)..];

        // Check if any recent experiment shows significant improvement
        let best_recent = recent_experiments
            .iter()
            .map(|e| {
                e.properties
                    .get("accuracy")
                    .and_then(|v| v.as_f64())
                    .or_else(|| {
                        e.properties
                            .get("loss")
                            .and_then(|v| v.as_f64())
                            .map(|v| -v)
                    })
                    .unwrap_or(0.0)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let older_experiments = &experiments[0..experiments.len().saturating_sub(config.patience)];
        let best_older = if older_experiments.is_empty() {
            f64::NEG_INFINITY
        } else {
            older_experiments
                .iter()
                .map(|e| {
                    e.properties
                        .get("accuracy")
                        .and_then(|v| v.as_f64())
                        .or_else(|| {
                            e.properties
                                .get("loss")
                                .and_then(|v| v.as_f64())
                                .map(|v| -v)
                        })
                        .unwrap_or(0.0)
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f64::NEG_INFINITY)
        };

        best_recent - best_older < config.min_improvement
    }
}

impl Default for ExperimentPipelineConfig {
    fn default() -> Self {
        Self {
            name: "default_pipeline".to_string(),
            max_experiments: 10,
            parallel_limit: 1,
            early_stopping: None,
        }
    }
}
