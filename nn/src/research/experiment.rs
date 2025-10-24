//! Unified Experiment Specification and Results
//!
//! This module defines standardized formats for experiment specifications,
//! execution results, and resource usage tracking across different research agents.

use std::collections::HashMap;
use std::time::Instant;

// Add required exports to parent module
pub use super::agent::ResourceRequirements;

use crate::error::{NNError, Result};

use super::{ResearchDomain, ResearchInsight};

/// Unified experiment specification
#[derive(Debug, Clone)]
pub struct ExperimentSpec {
    /// Unique experiment identifier
    pub id: String,
    /// Experiment name/description
    pub name: String,
    /// Target research domain
    pub domain: ResearchDomain,
    /// Agent that will execute this experiment
    pub agent_type: String,
    /// Experiment configuration (JSON)
    pub experiment_config: serde_json::Value,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Dependencies on other experiments
    pub dependencies: Vec<String>,
    /// Priority level (higher = more important)
    pub priority: u32,
    /// Timeout in seconds
    pub timeout_secs: Option<u64>,
    /// Quality constraints
    pub quality_constraints: QualityConstraints,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ExperimentSpec {
    /// Create new experiment specification
    pub fn new(id: String, name: String, domain: ResearchDomain, agent_type: String) -> Self {
        Self {
            id,
            name,
            domain,
            agent_type,
            experiment_config: serde_json::Value::Null,
            resource_requirements: ResourceRequirements::default(),
            dependencies: Vec::new(),
            priority: 1,
            timeout_secs: None,
            quality_constraints: QualityConstraints::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set experiment configuration
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.experiment_config = config;
        self
    }

    /// Set resource requirements
    pub fn with_resources(mut self, resources: ResourceRequirements) -> Self {
        self.resource_requirements = resources;
        self
    }

    /// Add dependency
    pub fn with_dependency(mut self, experiment_id: String) -> Self {
        self.dependencies.push(experiment_id);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Check if experiment dependencies are satisfied
    pub fn dependencies_satisfied(&self, completed_experiments: &[&str]) -> bool {
        self.dependencies.iter().all(|dep| completed_experiments.contains(&dep.as_str()))
    }

    /// Validate experiment specification
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "Experiment ID cannot be empty".to_string(),
            });
        }

        if self.name.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "Experiment name cannot be empty".to_string(),
            });
        }

        // Validate resource requirements
        if self.resource_requirements.cpu_cores == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "Resource requirements must specify at least 1 CPU core".to_string(),
            });
        }

        Ok(())
    }
}

/// Quality constraints for experiments
#[derive(Debug, Clone)]
pub struct QualityConstraints {
    /// Minimum performance threshold
    pub min_performance: Option<f64>,
    /// Maximum allowed variance
    pub max_variance: Option<f64>,
    /// Required statistical significance
    pub significance_level: Option<f64>,
    /// Minimum sample size for statistical tests
    pub min_sample_size: Option<usize>,
    /// Custom quality metrics
    pub custom_metrics: HashMap<String, QualityMetric>,
}

impl Default for QualityConstraints {
    fn default() -> Self {
        Self {
            min_performance: None,
            max_variance: None,
            significance_level: Some(0.05), // 95% confidence
            min_sample_size: Some(30),
            custom_metrics: HashMap::new(),
        }
    }
}

/// Individual quality metric
#[derive(Debug, Clone)]
pub struct QualityMetric {
    /// Metric name
    pub name: String,
    /// Target value
    pub target_value: f64,
    /// Tolerance (acceptable deviation)
    pub tolerance: f64,
    /// Comparison operator
    pub operator: QualityOperator,
}

/// Quality comparison operator
#[derive(Debug, Clone, PartialEq)]
pub enum QualityOperator {
    /// Greater than or equal
    GreaterEqual,
    /// Less than or equal
    LessEqual,
    /// Equal within tolerance
    Equal,
    /// Not equal (outside tolerance)
    NotEqual,
}

/// Experiment execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Experiment is pending execution
    Pending,
    /// Experiment is currently running
    Running,
    /// Experiment completed successfully
    Completed,
    /// Experiment failed
    Failed,
    /// Experiment was cancelled
    Cancelled,
    /// Experiment timed out
    Timeout,
}

/// Experiment execution result
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Experiment identifier
    pub experiment_id: String,
    /// Agent that executed the experiment
    pub agent_id: String,
    /// Execution status
    pub status: ExperimentStatus,
    /// Final performance metric
    pub final_performance: f64,
    /// Performance trajectory (if available)
    pub performance_trajectory: Vec<f64>,
    /// Resource usage statistics
    pub resource_usage: ResourceUsage,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Instant,
    /// Statistical analysis results
    pub statistics: ExperimentStatistics,
    /// Generated insights for knowledge transfer
    pub insights: Vec<ResearchInsight>,
    /// Experiment artifacts (serialized data)
    pub artifacts: HashMap<String, serde_json::Value>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ExperimentResult {
    /// Create new experiment result
    pub fn new(experiment_id: String, agent_id: String) -> Self {
        let now = Instant::now();
        Self {
            experiment_id,
            agent_id,
            status: ExperimentStatus::Pending,
            final_performance: 0.0,
            performance_trajectory: Vec::new(),
            resource_usage: ResourceUsage::default(),
            start_time: now,
            end_time: now,
            statistics: ExperimentStatistics::default(),
            insights: Vec::new(),
            artifacts: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Mark experiment as started
    pub fn mark_started(&mut self) {
        self.status = ExperimentStatus::Running;
        self.start_time = Instant::now();
    }

    /// Mark experiment as completed
    pub fn mark_completed(&mut self, final_performance: f64) {
        self.status = ExperimentStatus::Completed;
        self.final_performance = final_performance;
        self.end_time = Instant::now();
    }

    /// Mark experiment as failed
    pub fn mark_failed(&mut self, error_message: String) {
        self.status = ExperimentStatus::Failed;
        self.end_time = Instant::now();
        self.metadata.insert("error_message".to_string(), error_message);
    }

    /// Get execution duration
    pub fn duration(&self) -> std::time::Duration {
        self.end_time.duration_since(self.start_time)
    }

    /// Check if experiment meets quality constraints
    pub fn meets_quality_constraints(&self, constraints: &QualityConstraints) -> bool {
        // Check minimum performance
        if let Some(min_perf) = constraints.min_performance {
            if self.final_performance < min_perf {
                return false;
            }
        }

        // Check maximum variance
        if let Some(max_var) = constraints.max_variance {
            if let Some(variance) = self.statistics.variance {
                if variance > max_var {
                    return false;
                }
            }
        }

        // Check statistical significance
        if let Some(sig_level) = constraints.significance_level {
            if let Some(p_value) = self.statistics.p_value {
                if p_value > sig_level {
                    return false;
                }
            }
        }

        // Check custom metrics
        for (metric_name, quality_metric) in &constraints.custom_metrics {
            if let Some(actual_value) = self.metadata.get(metric_name)
                .and_then(|v| v.parse::<f64>().ok()) {

                let meets_constraint = match quality_metric.operator {
                    QualityOperator::GreaterEqual => actual_value >= quality_metric.target_value,
                    QualityOperator::LessEqual => actual_value <= quality_metric.target_value,
                    QualityOperator::Equal => (actual_value - quality_metric.target_value).abs() <= quality_metric.tolerance,
                    QualityOperator::NotEqual => (actual_value - quality_metric.target_value).abs() > quality_metric.tolerance,
                };

                if !meets_constraint {
                    return false;
                }
            }
        }

        true
    }

    /// Add performance measurement
    pub fn add_performance_measurement(&mut self, performance: f64) {
        self.performance_trajectory.push(performance);
        self.final_performance = performance;
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        format!(
            "Experiment {} (Agent: {}): Status={:?}, Performance={:.4}, Duration={:.2}s",
            self.experiment_id,
            self.agent_id,
            self.status,
            self.final_performance,
            self.duration().as_secs_f64()
        )
    }
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU time used (seconds)
    pub cpu_time_secs: f64,
    /// Peak CPU usage (%)
    pub peak_cpu_usage: f64,
    /// GPU time used (seconds)
    pub gpu_time_secs: f64,
    /// Peak GPU memory usage (GB)
    pub peak_gpu_memory_gb: f64,
    /// Peak system memory usage (GB)
    pub peak_system_memory_gb: f64,
    /// Storage space used (GB)
    pub storage_used_gb: f64,
    /// Network usage (MB)
    pub network_usage_mb: f64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_time_secs: 0.0,
            peak_cpu_usage: 0.0,
            gpu_time_secs: 0.0,
            peak_gpu_memory_gb: 0.0,
            peak_system_memory_gb: 0.0,
            storage_used_gb: 0.0,
            network_usage_mb: 0.0,
        }
    }
}

/// Statistical analysis results
#[derive(Debug, Clone)]
pub struct ExperimentStatistics {
    /// Mean performance
    pub mean: Option<f64>,
    /// Standard deviation
    pub std_dev: Option<f64>,
    /// Variance
    pub variance: Option<f64>,
    /// Confidence interval (95%)
    pub confidence_interval: Option<(f64, f64)>,
    /// P-value for statistical tests
    pub p_value: Option<f64>,
    /// Effect size
    pub effect_size: Option<f64>,
    /// Statistical power
    pub statistical_power: Option<f64>,
    /// Sample size used
    pub sample_size: usize,
    /// Distribution characteristics
    pub distribution: Option<DistributionType>,
}

impl Default for ExperimentStatistics {
    fn default() -> Self {
        Self {
            mean: None,
            std_dev: None,
            variance: None,
            confidence_interval: None,
            p_value: None,
            effect_size: None,
            statistical_power: None,
            sample_size: 0,
            distribution: None,
        }
    }
}

/// Distribution type for statistical analysis
#[derive(Debug, Clone)]
pub enum DistributionType {
    /// Normal distribution
    Normal,
    /// Log-normal distribution
    LogNormal,
    /// Student's t-distribution
    StudentT { degrees_of_freedom: usize },
    /// Custom distribution with parameters
    Custom(HashMap<String, f64>),
}

/// Experiment batch for parallel execution
#[derive(Debug, Clone)]
pub struct ExperimentBatch {
    /// Batch identifier
    pub id: String,
    /// Experiments in this batch
    pub experiments: Vec<ExperimentSpec>,
    /// Required resources for entire batch
    pub batch_resources: ResourceRequirements,
    /// Maximum parallel executions
    pub max_parallel: usize,
    /// Execution strategy
    pub strategy: BatchExecutionStrategy,
}

impl ExperimentBatch {
    /// Create new experiment batch
    pub fn new(id: String) -> Self {
        Self {
            id,
            experiments: Vec::new(),
            batch_resources: ResourceRequirements::default(),
            max_parallel: 4,
            strategy: BatchExecutionStrategy::Parallel,
        }
    }

    /// Add experiment to batch
    pub fn add_experiment(&mut self, experiment: ExperimentSpec) {
        self.experiments.push(experiment);
        // Update batch resource requirements
        self.batch_resources.cpu_cores = self.batch_resources.cpu_cores.max(
            self.experiments.iter().map(|e| e.resource_requirements.cpu_cores).max().unwrap_or(1)
        );
        self.batch_resources.gpu_memory_gb = self.batch_resources.gpu_memory_gb.max(
            self.experiments.iter().map(|e| e.resource_requirements.gpu_memory_gb).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0)
        );
    }

    /// Validate batch configuration
    pub fn validate(&self) -> Result<()> {
        if self.experiments.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "Experiment batch cannot be empty".to_string(),
            });
        }

        for experiment in &self.experiments {
            experiment.validate()?;
        }

        Ok(())
    }
}

/// Batch execution strategy
#[derive(Debug, Clone)]
pub enum BatchExecutionStrategy {
    /// Execute experiments in parallel
    Parallel,
    /// Execute experiments sequentially
    Sequential,
    /// Execute with dependencies
    WithDependencies,
    /// Adaptive execution based on resource availability
    Adaptive,
}

/// Experiment execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Available resources
    pub available_resources: ResourceRequirements,
    /// Time budget remaining
    pub time_budget_secs: u64,
    /// Quality requirements
    pub quality_threshold: f64,
    /// Previous experiment results
    pub previous_results: Vec<ExperimentResult>,
    /// Environment variables
    pub environment: HashMap<String, String>,
}

impl ExecutionContext {
    /// Check if resources are sufficient for experiment
    pub fn has_sufficient_resources(&self, experiment: &ExperimentSpec) -> bool {
        let req = &experiment.resource_requirements;

        self.available_resources.cpu_cores >= req.cpu_cores &&
        self.available_resources.gpu_memory_gb >= req.gpu_memory_gb &&
        self.available_resources.system_memory_gb >= req.system_memory_gb &&
        self.available_resources.storage_gb >= req.storage_gb
    }

    /// Check if time budget allows experiment execution
    pub fn within_time_budget(&self, experiment: &ExperimentSpec) -> bool {
        let estimated_time = experiment.timeout_secs.unwrap_or(3600);
        estimated_time <= self.time_budget_secs
    }
}

/// Experiment queue for scheduling
#[derive(Debug)]
pub struct ExperimentQueue {
    /// Pending experiments
    pending: Vec<ExperimentSpec>,
    /// Running experiments
    running: HashMap<String, ExperimentSpec>,
    /// Completed experiments
    completed: HashMap<String, ExperimentResult>,
}

impl ExperimentQueue {
    /// Create new experiment queue
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            running: HashMap::new(),
            completed: HashMap::new(),
        }
    }

    /// Add experiment to queue
    pub fn enqueue(&mut self, experiment: ExperimentSpec) {
        self.pending.push(experiment);
        // Sort by priority (higher priority first)
        self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get next experiment that can be executed
    pub fn dequeue(&mut self, context: &ExecutionContext) -> Option<ExperimentSpec> {
        // Find first experiment that can be executed
        let mut index_to_remove = None;

        for (i, experiment) in self.pending.iter().enumerate() {
            if experiment.dependencies_satisfied(
                &self.completed.keys().map(|s| s.as_str()).collect::<Vec<_>>()
            ) && context.has_sufficient_resources(experiment)
            && context.within_time_budget(experiment) {
                index_to_remove = Some(i);
                break;
            }
        }

        if let Some(index) = index_to_remove {
            let experiment = self.pending.remove(index);
            self.running.insert(experiment.id.clone(), experiment.clone());
            Some(experiment)
        } else {
            None
        }
    }

    /// Mark experiment as completed
    pub fn complete(&mut self, experiment_id: String, result: ExperimentResult) {
        self.running.remove(&experiment_id);
        self.completed.insert(experiment_id, result);
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStatistics {
        QueueStatistics {
            pending_count: self.pending.len(),
            running_count: self.running.len(),
            completed_count: self.completed.len(),
            avg_wait_time: 0.0, // Would need timestamp tracking
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Number of pending experiments
    pub pending_count: usize,
    /// Number of running experiments
    pub running_count: usize,
    /// Number of completed experiments
    pub completed_count: usize,
    /// Average wait time (seconds)
    pub avg_wait_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_spec_validation() {
        let mut spec = ExperimentSpec::new(
            "test_exp".to_string(),
            "Test Experiment".to_string(),
            ResearchDomain::GeneralML,
            "test_agent".to_string(),
        );

        // Should validate successfully
        assert!(spec.validate().is_ok());

        // Test invalid spec
        let invalid_spec = ExperimentSpec {
            id: "".to_string(),
            ..spec
        };

        assert!(invalid_spec.validate().is_err());
    }

    #[test]
    fn test_experiment_result() {
        let mut result = ExperimentResult::new("exp1".to_string(), "agent1".to_string());

        result.mark_started();
        std::thread::sleep(std::time::Duration::from_millis(10));
        result.mark_completed(0.95);

        assert_eq!(result.status, ExperimentStatus::Completed);
        assert_eq!(result.final_performance, 0.95);
        assert!(result.duration().as_millis() >= 10);
    }

    #[test]
    fn test_experiment_queue() {
        let mut queue = ExperimentQueue::new();

        let exp1 = ExperimentSpec::new(
            "exp1".to_string(),
            "Experiment 1".to_string(),
            ResearchDomain::GeneralML,
            "agent1".to_string(),
        ).with_priority(2);

        let exp2 = ExperimentSpec::new(
            "exp2".to_string(),
            "Experiment 2".to_string(),
            ResearchDomain::GeneralML,
            "agent1".to_string(),
        ).with_priority(1);

        queue.enqueue(exp2);
        queue.enqueue(exp1);

        let context = ExecutionContext {
            available_resources: ResourceRequirements::default(),
            time_budget_secs: 3600,
            quality_threshold: 0.8,
            previous_results: Vec::new(),
            environment: HashMap::new(),
        };

        // Higher priority should come first
        let next = queue.dequeue(&context).unwrap();
        assert_eq!(next.id, "exp1"); // Priority 2

        let next2 = queue.dequeue(&context).unwrap();
        assert_eq!(next2.id, "exp2"); // Priority 1
    }
}
