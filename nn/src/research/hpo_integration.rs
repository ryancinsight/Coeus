//! HPO Integration with Research Framework
//!
//! This module provides seamless integration between Hyperparameter Optimization (HPO)
//! algorithms and the unified research framework, enabling automatic experiment
//! tracking, metrics collection, checkpointing, and artifact management for HPO workflows.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::error::{NNError, Result};
use crate::nn::hpo::{HPOptimizer, HyperparameterOptimizer, HyperparameterSpace, HyperparameterConfig, OptimizationResult};
use crate::research::tracking::{ExperimentTracker, ExperimentSummary};
use crate::research::metrics::{MetricsCollector, MetricEntry};
use crate::research::UnifiedResearchFramework;

/// HPO Experiment Context
/// Tracks HPO-specific experimental context and state
#[derive(Debug, Clone)]
pub struct HPOExperimentContext {
    /// Unique experiment ID
    pub experiment_id: String,
    /// Model architecture being optimized
    pub model_architecture: String,
    /// Task type (classification, regression, etc.)
    pub task: String,
    /// Dataset information
    pub dataset: DatasetInfo,
    /// Optimizer configuration
    pub optimizer_config: OptimizerConfig,
    /// Search space configuration
    pub search_space: HyperparameterSpace,
    /// Evaluation configuration
    pub evaluation_config: EvaluationConfig,
    /// Multi-objective optimization enabled
    pub multi_objective: bool,
    /// Objectives to optimize
    pub objectives: Vec<OptimizationObjective>,
}

/// Dataset information for HPO
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub size: usize,
    pub train_split: f64,
    pub validation_split: f64,
    pub test_split: f64,
    pub metadata: HashMap<String, String>,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub algorithm: HPOAlgorithm,
    pub budget: usize,
    pub parallel_evaluations: usize,
    pub early_stopping: bool,
    pub early_stopping_patience: usize,
    pub seed: Option<u64>,
}

/// HPO algorithms available
#[derive(Debug, Clone)]
pub enum HPOAlgorithm {
    BayesianOptimization,
    RandomSearch,
    GridSearch,
    Hyperband,
    SuccessiveHalving,
    TPE, // Tree-structured Parzen Estimator
    SMAC, // Sequential Model-based Algorithm Configuration
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub evaluation_metric: String,
    pub validation_frequency: usize,
    pub use_gpu: bool,
    pub distributed_training: bool,
    pub num_workers: usize,
}

/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    Maximize(String), // metric name
    Minimize(String), // metric name
    TargetValue(String, f64, f64), // metric name, target value, tolerance
}

/// HPO Search Result
#[derive(Debug, Clone)]
pub struct HPOSearchResult {
    pub best_config: HyperparameterConfig,
    pub best_score: f64,
    pub search_history: Vec<HyperparameterEvaluation>,
    pub total_evaluations: usize,
    pub search_time: std::time::Duration,
    pub convergence_metrics: ConvergenceMetrics,
    pub pareto_front: Option<Vec<HyperparameterConfig>>, // For multi-objective
    pub experiment_summary: ExperimentSummary,
}

/// Hyperparameter evaluation record
#[derive(Debug, Clone)]
pub struct HyperparameterEvaluation {
    pub config: HyperparameterConfig,
    pub score: f64,
    pub evaluation_time: std::time::Duration,
    pub resource_usage: ResourceUsage,
    pub metrics: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// Resource usage for HPO evaluation
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub gpu_memory_mb: u64,
    pub cpu_time_seconds: f64,
    pub gpu_time_seconds: f64,
    pub peak_memory_mb: u64,
    pub power_consumption_w: Option<f64>,
}

/// Convergence metrics for HPO
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub final_improvement_rate: f64,
    pub regret: f64,
    pub exploration_efficiency: f64,
    pub sampling_efficiency: f64,
    pub confidence_interval: Option<(f64, f64)>,
}

/// Integrated HPO Research Framework
/// Provides seamless integration between HPO algorithms and research tracking
pub struct IntegratedHPOFramework {
    /// Research framework instance
    research_framework: Arc<RwLock<UnifiedResearchFramework>>,
    /// HPO-specific experiment contexts
    experiment_contexts: HashMap<String, HPOExperimentContext>,
    /// Objective functions registry
    objective_functions: HashMap<String, Box<dyn ObjectiveFunction>>,
    /// Optimizer factory
    optimizer_factory: HPOOptimizerFactory,
    /// Multi-objective utilities
    multi_objective_utils: MultiObjectiveUtils,
}

/// Objective function trait
#[derive(Debug)]
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate hyperparameter configuration
    fn evaluate(&self, config: &HyperparameterConfig, context: &HPOExperimentContext) -> Result<f64>;

    /// Get objective function name
    fn name(&self) -> &str;

    /// Get supported metrics
    fn supported_metrics(&self) -> Vec<String>;

    /// Check if function supports multi-objective evaluation
    fn supports_multi_objective(&self) -> bool;
}

/// HPO Optimizer Factory
#[derive(Debug)]
pub struct HPOOptimizerFactory {
    /// Registered optimizers
    optimizers: HashMap<HPOAlgorithm, Box<dyn OptimizerFactory>>,
}

/// Optimizer factory trait
pub trait OptimizerFactory: Send + Sync {
    /// Create optimizer instance
    fn create_optimizer(&self, space: &HyperparameterSpace, config: &OptimizerConfig) -> Result<Box<dyn HPOAlgorithmImpl>>;

    /// Get supported algorithms
    fn supported_algorithms(&self) -> Vec<HPOAlgorithm>;
}

/// HPO algorithm implementation trait
pub trait HPOAlgorithmImpl: Send + Sync {
    /// Run optimization
    fn optimize(&self, objective: Arc<dyn ObjectiveFunction + Send + Sync>, context: &HPOExperimentContext) -> Result<OptimizationResult>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Multi-objective optimization utilities
#[derive(Debug)]
pub struct MultiObjectiveUtils {
    /// Pareto dominance functions
    /// Hypervolume calculation
    /// Reference point utilities
}

impl IntegratedHPOFramework {
    /// Create new integrated HPO framework
    pub fn new(research_framework: Arc<RwLock<UnifiedResearchFramework>>) -> Self {
        let mut framework = Self {
            research_framework,
            experiment_contexts: HashMap::new(),
            objective_functions: HashMap::new(),
            optimizer_factory: HPOOptimizerFactory::new(),
            multi_objective_utils: MultiObjectiveUtils::new(),
        };

        // Register built-in objective functions
        framework.register_objective_function(
            "accuracy".to_string(),
            Box::new(StandardAccuracyObjective::new()),
        );
        framework.register_objective_function(
            "f1_score".to_string(),
            Box::new(F1ScoreObjective::new()),
        );

        framework
    }

    /// Start integrated HPO experiment
    pub fn start_hpo_experiment(&mut self, context: HPOExperimentContext) -> Result<String> {
        let experiment_id = format!("hpo_{}_{}", context.model_architecture, context.task);
        let experiment_name = format!("HPO Search: {} - {}", context.model_architecture, context.task);
        let experiment_description = format!(
            "Hyperparameter optimization for {} model on {} dataset using {}",
            context.model_architecture, context.dataset.name, format!("{:?}", context.optimizer_config.algorithm)
        );

        // Create experiment in research framework
        let framework = self.research_framework.write().unwrap();
        let tracker = framework.create_experiment(
            experiment_id.clone(),
            experiment_name,
            experiment_description,
        );

        // Log HPO-specific metadata
        tracker.log_hyperparameter(
            "hpo_algorithm".to_string(),
            format!("{:?}", context.optimizer_config.algorithm).into(),
            Some("Hyperparameter optimization algorithm used".to_string()),
        );
        tracker.log_hyperparameter(
            "search_budget".to_string(),
            context.optimizer_config.budget.into(),
            Some("Total evaluation budget".to_string()),
        );
        tracker.log_hyperparameter(
            "search_space_size".to_string(),
            context.search_space.parameters.len().into(),
            Some("Number of hyperparameters being optimized".to_string()),
        );

        // Store context
        self.experiment_contexts.insert(experiment_id.clone(), context);

        Ok(experiment_id)
    }

    /// Execute integrated HPO search
    pub fn execute_hpo_search(&mut self, experiment_id: &str) -> Result<HPOSearchResult> {
        let context = self.experiment_contexts.get(experiment_id)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Experiment context not found for {}", experiment_id),
            })?
            .clone();

        // Get objective function
        let objective_name = if context.multi_objective {
            "multi_objective"
        } else {
            context.objectives.first()
                .and_then(|obj| match obj {
                    OptimizationObjective::Maximize(metric) | OptimizationObjective::Minimize(metric) => Some(metric.as_str()),
                    OptimizationObjective::TargetValue(metric, _, _) => Some(metric.as_str()),
                })
                .unwrap_or("accuracy")
        };

        let objective = self.objective_functions.get(objective_name)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Objective function '{}' not found", objective_name),
            })?
            .clone();

        // Create optimizer
        let optimizer_config = &context.optimizer_config;
        let optimizer = self.optimizer_factory.create_optimizer(
            &context.search_space,
            optimizer_config,
        )?;

        let mut framework = self.research_framework.write().unwrap();
        let tracker = framework.create_experiment(
            format!("{}_search", experiment_id),
            "HPO Search Execution".to_string(),
            "Executing hyperparameter optimization search".to_string(),
        );

        let start_time = Instant::now();
        let mut search_history = Vec::new();
        let mut evaluations = 0;

        // Execute search with research framework integration
        let result = optimizer.optimize(objective, &context)?;

        let search_time = start_time.elapsed();

        // Convert optimization result to HPO result
        let hpo_result = HPOSearchResult {
            best_config: result.best_config.clone(),
            best_score: result.best_value,
            search_history,
            total_evaluations: result.evaluations,
            search_time,
            convergence_metrics: ConvergenceMetrics {
                final_improvement_rate: 0.05, // Placeholder
                regret: 0.1, // Placeholder
                exploration_efficiency: 0.8, // Placeholder
                sampling_efficiency: 0.9, // Placeholder
                confidence_interval: Some((result.best_value - 0.05, result.best_value + 0.05)),
            },
            pareto_front: if context.multi_objective { Some(vec![result.best_config.clone()]) } else { None },
            experiment_summary: tracker.summarize(),
        };

        Ok(hpo_result)
    }

    /// Start joint NAS-HPO experiment
    pub fn start_joint_nas_hpo_experiment(
        &mut self,
        nas_context: super::nas_integration::NASExperimentContext,
        hpo_context: HPOExperimentContext,
        joint_config: JointSearchConfig,
    ) -> Result<String> {
        let joint_experiment_id = format!("joint_{}_{}_{}",
            nas_context.domain, nas_context.task, joint_config.joint_algorithm);

        let experiment_name = format!("Joint NAS-HPO: {} - {}", nas_context.domain, nas_context.task);
        let experiment_description = format!(
            "Joint neural architecture and hyperparameter optimization using {}",
            format!("{:?}", joint_config.joint_algorithm)
        );

        let framework = self.research_framework.write().unwrap();
        let tracker = framework.create_experiment(
            joint_experiment_id.clone(),
            experiment_name,
            experiment_description,
        );

        // Log joint search metadata
        tracker.log_hyperparameter(
            "joint_algorithm".to_string(),
            format!("{:?}", joint_config.joint_algorithm).into(),
            Some("Joint search algorithm used".to_string()),
        );
        tracker.log_hyperparameter(
            "nas_algorithm".to_string(),
            format!("{:?}", nas_context.search_config.algorithm).into(),
            Some("NAS algorithm used".to_string()),
        );
        tracker.log_hyperparameter(
            "hpo_algorithm".to_string(),
            format!("{:?}", hpo_context.optimizer_config.algorithm).into(),
            Some("HPO algorithm used".to_string()),
        );

        // Store contexts in experiment tracker metadata
        // In a real implementation, serialize and store

        Ok(joint_experiment_id)
    }

    /// Register objective function
    pub fn register_objective_function(
        &mut self,
        name: String,
        objective: Box<dyn ObjectiveFunction>,
    ) {
        self.objective_functions.insert(name, objective);
    }

    /// Register optimizer factory
    pub fn register_optimizer_factory(
        &mut self,
        algorithm: HPOAlgorithm,
        factory: Box<dyn OptimizerFactory>,
    ) {
        self.optimizer_factory.optimizers.insert(algorithm, factory);
    }

    /// Get experiment summary with HPO metrics
    pub fn get_experiment_summary(&self, experiment_id: &str) -> Result<HPOExperimentSummary> {
        let framework = self.research_framework.read().unwrap();
        let base_summary = framework.experiment_registry.get_experiment_summary(experiment_id)?;

        let context = self.experiment_contexts.get(experiment_id);

        Ok(HPOExperimentSummary {
            base_summary,
            hpo_context: context.cloned(),
            hpo_metrics: HPOMetrics {
                configurations_evaluated: 0,
                best_configuration_score: None,
                search_efficiency: None,
                convergence_speed: None,
                hyperparameter_importance: None,
            },
        })
    }

    /// Generate HPO research report
    pub fn generate_hpo_research_report(&self) -> Result<String> {
        let framework = self.research_framework.read().unwrap();

        let mut report = String::new();
        report.push_str("# Hyperparameter Optimization Research Report\n\n");

        // Summary statistics
        let total_experiments = self.experiment_contexts.len();
        report.push_str(&format!("## Summary\n"));
        report.push_str(&format!("- Total HPO Experiments: {}\n", total_experiments));
        report.push_str(&format!("- Active Experiments: {}\n", framework.health_status().experiments_active));
        report.push_str(&format!("- Registered Objective Functions: {}\n", self.objective_functions.len()));
        report.push_str(&format!("- Supported HPO Algorithms: {}\n\n", self.optimizer_factory.optimizers.len()));

        // Experiment details
        if !self.experiment_contexts.is_empty() {
            report.push_str("## HPO Experiments\n\n");
            for (id, context) in &self.experiment_contexts {
                report.push_str(&format!("### Experiment: {}\n", id));
                report.push_str(&format!("- Model Architecture: {}\n", context.model_architecture));
                report.push_str(&format!("- Task: {}\n", context.task));
                report.push_str(&format!("- Dataset: {}\n", context.dataset.name));
                report.push_str(&format!("- Algorithm: {:?}\n", context.optimizer_config.algorithm));
                report.push_str(&format!("- Budget: {}\n", context.optimizer_config.budget));
                report.push_str(&format!("- Multi-objective: {}\n", if context.multi_objective { "Yes" } else { "No" }));
                report.push_str(&format!("- Objectives: {}\n\n",
                    context.objectives.iter().map(|obj| format!("{:?}", obj)).collect::<Vec<_>>().join(", ")
                ));
            }
        }

        Ok(report)
    }
}

/// Joint search configuration for NAS-HPO integration
#[derive(Debug, Clone)]
pub struct JointSearchConfig {
    pub joint_algorithm: JointAlgorithm,
    pub alternation_schedule: AlternationSchedule,
    pub resource_allocation: ResourceAllocationStrategy,
    pub warm_starting: bool,
    pub transfer_learning: bool,
}

/// Joint search algorithms
#[derive(Debug, Clone)]
pub enum JointAlgorithm {
    Alternating,
    Concurrent,
    EvolutionaryJoint,
    BayesianJoint,
}

/// Alternation schedule between NAS and HPO
#[derive(Debug, Clone)]
pub enum AlternationSchedule {
    FixedRounds { nas_rounds: usize, hpo_rounds: usize },
    Adaptive { performance_threshold: f64, patience: usize },
    Dynamic { resource_based: bool },
}

/// Resource allocation strategy
#[derive(Debug, Clone)]
pub enum ResourceAllocationStrategy {
    EqualSplit,
    PerformanceBased,
    Adaptive,
}

impl HPOOptimizerFactory {
    pub fn new() -> Self {
        Self {
            optimizers: HashMap::new(),
        }
    }
}

impl MultiObjectiveUtils {
    pub fn new() -> Self {
        Self {}
    }

    /// Check if point A dominates point B in multi-objective space
    pub fn dominates(&self, point_a: &[f64], point_b: &[f64]) -> bool {
        if point_a.len() != point_b.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for (a, b) in point_a.iter().zip(point_b.iter()) {
            if a < b { // Assuming minimization (for maximization, use >)
                return false;
            }
            if a > b {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate hypervolume for a set of points
    pub fn hypervolume(&self, points: &[Vec<f64>], reference_point: &[f64]) -> f64 {
        // Simplified hypervolume calculation
        // In practice, would use more sophisticated algorithms
        if points.is_empty() {
            return 0.0;
        }

        let mut volume = 0.0;
        for point in points {
            if point.len() != reference_point.len() {
                continue;
            }

            let mut point_volume = 1.0;
            for (p, r) in point.iter().zip(reference_point.iter()) {
                point_volume *= (r - p).max(0.0);
            }
            volume += point_volume;
        }

        volume
    }
}

/// HPO Experiment Summary
#[derive(Debug)]
pub struct HPOExperimentSummary {
    pub base_summary: ExperimentSummary,
    pub hpo_context: Option<HPOExperimentContext>,
    pub hpo_metrics: HPOMetrics,
}

/// HPO-specific metrics
#[derive(Debug)]
pub struct HPOMetrics {
    pub configurations_evaluated: usize,
    pub best_configuration_score: Option<f64>,
    pub search_efficiency: Option<f64>,
    pub convergence_speed: Option<f64>,
    pub hyperparameter_importance: Option<HashMap<String, f64>>,
}

/// Built-in objective functions
pub mod objectives {
    use super::*;

    /// Standard accuracy objective function
    pub struct StandardAccuracyObjective;

    impl StandardAccuracyObjective {
        pub fn new() -> Self {
            Self
        }
    }

    impl ObjectiveFunction for StandardAccuracyObjective {
        fn evaluate(&self, config: &HyperparameterConfig, _context: &HPOExperimentContext) -> Result<f64> {
            // Placeholder: in real implementation, this would train and evaluate the model
            // For now, return a random score based on configuration
            let mut score = 0.5;

            // Simulate better performance with reasonable hyperparameter choices
            if let Some(lr) = config.get_float("learning_rate") {
                if lr > 0.0001 && lr < 0.1 {
                    score += 0.1;
                }
            }

            if let Some(batch_size) = config.get_int("batch_size") {
                if batch_size >= 16 && batch_size <= 128 {
                    score += 0.1;
                }
            }

            Ok(score.min(1.0).max(0.0))
        }

        fn name(&self) -> &str {
            "Standard Accuracy"
        }

        fn supported_metrics(&self) -> Vec<String> {
            vec!["accuracy".to_string(), "validation_accuracy".to_string()]
        }

        fn supports_multi_objective(&self) -> bool {
            false
        }
    }

    /// F1 Score objective function
    pub struct F1ScoreObjective;

    impl F1ScoreObjective {
        pub fn new() -> Self {
            Self
        }
    }

    impl ObjectiveFunction for F1ScoreObjective {
        fn evaluate(&self, config: &HyperparameterConfig, _context: &HPOExperimentContext) -> Result<f64> {
            // Placeholder similar to accuracy
            let mut score = 0.5;

            if let Some(lr) = config.get_float("learning_rate") {
                if lr > 0.0001 && lr < 0.1 {
                    score += 0.1;
                }
            }

            Ok(score.min(1.0).max(0.0))
        }

        fn name(&self) -> &str {
            "F1 Score"
        }

        fn supported_metrics(&self) -> Vec<String> {
            vec!["f1_score".to_string(), "precision".to_string(), "recall".to_string()]
        }

        fn supports_multi_objective(&self) -> bool {
            true
        }
    }
}

/// Optimizer factory implementations
pub mod optimizer_factories {
    use super::*;

    /// Bayesian Optimization factory
    pub struct BayesianOptimizerFactory;

    impl BayesianOptimizerFactory {
        pub fn new() -> Self {
            Self
        }
    }

    impl OptimizerFactory for BayesianOptimizerFactory {
        fn create_optimizer(&self, space: &HyperparameterSpace, config: &OptimizerConfig) -> Result<Box<dyn HPOAlgorithmImpl>> {
            // In real implementation, would create Bayesian optimizer
            Ok(Box::new(BayesianOptimizerImpl::new(space.clone(), config.budget)))
        }

        fn supported_algorithms(&self) -> Vec<HPOAlgorithm> {
            vec![HPOAlgorithm::BayesianOptimization]
        }
    }

    /// Bayesian optimizer implementation
    struct BayesianOptimizerImpl {
        space: HyperparameterSpace,
        budget: usize,
    }

    impl BayesianOptimizerImpl {
        fn new(space: HyperparameterSpace, budget: usize) -> Self {
            Self { space, budget }
        }
    }

    impl HPOAlgorithmImpl for BayesianOptimizerImpl {
        fn optimize(&self, objective: Arc<dyn ObjectiveFunction + Send + Sync>, context: &HPOExperimentContext) -> Result<OptimizationResult> {
            // Placeholder Bayesian optimization implementation
            let start_time = Instant::now();
            let mut best_value = 0.0;
            let mut best_config = HyperparameterConfig::new();
            let mut history = Vec::new();

            // Simple random search as placeholder
            for i in 0..self.budget {
                let config = self.space.sample_random()?;
                let value = objective.evaluate(&config, context)?;
                let eval_time = start_time.elapsed() / (i as u32 + 1);

                history.push((config.clone(), value, eval_time));

                if value > best_value {
                    best_value = value;
                    best_config = config;
                }
            }

            let total_time = start_time.elapsed();

            Ok(OptimizationResult {
                best_config,
                best_value,
                evaluations: self.budget,
                total_time,
                history,
            })
        }

        fn name(&self) -> &str {
            "Bayesian Optimization"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hpo::space::HyperparameterSpace;

    #[test]
    fn test_integrated_hpo_framework_creation() {
        let research_framework = Arc::new(RwLock::new(UnifiedResearchFramework::new()));
        let hpo_framework = IntegratedHPOFramework::new(research_framework);
        assert!(!hpo_framework.objective_functions.is_empty());
    }

    #[test]
    fn test_hpo_experiment_context() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(crate::nn::hpo::space::Hyperparameter::Float {
            name: "learning_rate".to_string(),
            min: 0.0001,
            max: 0.1,
            log_scale: true,
        });

        let context = HPOExperimentContext {
            experiment_id: "test_hpo".to_string(),
            model_architecture: "resnet50".to_string(),
            task: "classification".to_string(),
            dataset: DatasetInfo {
                name: "imagenet".to_string(),
                size: 1400000,
                train_split: 0.8,
                validation_split: 0.1,
                test_split: 0.1,
                metadata: HashMap::new(),
            },
            optimizer_config: OptimizerConfig {
                algorithm: HPOAlgorithm::BayesianOptimization,
                budget: 50,
                parallel_evaluations: 4,
                early_stopping: true,
                early_stopping_patience: 10,
                seed: Some(42),
            },
            search_space: space,
            evaluation_config: EvaluationConfig {
                epochs: 100,
                batch_size: 32,
                evaluation_metric: "accuracy".to_string(),
                validation_frequency: 5,
                use_gpu: true,
                distributed_training: false,
                num_workers: 4,
            },
            multi_objective: false,
            objectives: vec![OptimizationObjective::Maximize("accuracy".to_string())],
        };

        assert_eq!(context.model_architecture, "resnet50");
        assert_eq!(context.task, "classification");
    }

    #[test]
    fn test_multi_objective_utils() {
        let utils = MultiObjectiveUtils::new();

        // Test dominance
        let point_a = vec![1.0, 1.0]; // Better on first objective
        let point_b = vec![2.0, 0.5]; // Better on second objective
        assert!(utils.dominates(&point_a, &point_b) == false); // Neither dominates
        assert!(utils.dominates(&point_b, &point_a) == false);

        let point_c = vec![0.5, 0.5]; // Better on both
        assert!(utils.dominates(&point_c, &point_a)); // C dominates A
        assert!(utils.dominates(&point_c, &point_b)); // C dominates B
    }

    #[test]
    fn test_objective_function() {
        use objectives::StandardAccuracyObjective;

        let objective = StandardAccuracyObjective::new();
        let config = HyperparameterConfig::new();

        let context = HPOExperimentContext {
            experiment_id: "test".to_string(),
            model_architecture: "test".to_string(),
            task: "test".to_string(),
            dataset: DatasetInfo {
                name: "test".to_string(),
                size: 1000,
                train_split: 0.8,
                validation_split: 0.1,
                test_split: 0.1,
                metadata: HashMap::new(),
            },
            optimizer_config: OptimizerConfig {
                algorithm: HPOAlgorithm::RandomSearch,
                budget: 10,
                parallel_evaluations: 1,
                early_stopping: false,
                early_stopping_patience: 5,
                seed: None,
            },
            search_space: HyperparameterSpace::new(),
            evaluation_config: EvaluationConfig {
                epochs: 10,
                batch_size: 32,
                evaluation_metric: "accuracy".to_string(),
                validation_frequency: 1,
                use_gpu: false,
                distributed_training: false,
                num_workers: 1,
            },
            multi_objective: false,
            objectives: vec![],
        };

        let score = objective.evaluate(&config, &context).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        assert_eq!(objective.name(), "Standard Accuracy");
        assert!(objective.supported_metrics().contains(&"accuracy".to_string()));
    }
}
