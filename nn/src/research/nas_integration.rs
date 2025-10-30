//! NAS Integration with Research Framework
//!
//! This module provides seamless integration between Neural Architecture Search (NAS)
//! algorithms and the unified research framework, enabling automatic experiment
//! tracking, metrics collection, checkpointing, and artifact management for NAS workflows.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::error::{NNError, Result};
use crate::nn::nas::{Architecture, ArchitectureEvaluator, ArchitectureSpace, EvolutionaryNAS, ReinforcementNAS, DartsNAS, AutomatedResearchPipeline};
use crate::nn::hpo::{HPOptimizer, HyperparameterOptimizer};
use crate::research::tracking::{ExperimentTracker, ExperimentRegistry, ExperimentSummary};
use crate::research::metrics::{MetricsCollector, MetricEntry};
use crate::research::UnifiedResearchFramework;

/// NAS Experiment Context
/// Tracks NAS-specific experimental context and state
#[derive(Debug, Clone)]
pub struct NASExperimentContext {
    /// Unique experiment ID
    pub experiment_id: String,
    /// Research domain (computer_vision, nlp, etc.)
    pub domain: String,
    /// Task type (classification, regression, etc.)
    pub task: String,
    /// Dataset information
    pub dataset: DatasetInfo,
    /// Search space configuration
    pub search_space_config: SearchSpaceConfig,
    /// Search algorithm configuration
    pub search_config: SearchConfig,
    /// Performance prediction enabled
    pub performance_prediction: bool,
    /// Joint NAS-HPO enabled
    pub joint_search: bool,
}

/// Dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub size: usize,
    pub input_shape: Vec<usize>,
    pub output_classes: usize,
    pub metadata: HashMap<String, String>,
}

/// Search space configuration
#[derive(Debug, Clone)]
pub struct SearchSpaceConfig {
    pub max_layers: usize,
    pub available_operations: Vec<String>,
    pub parameter_ranges: HashMap<String, ParameterRange>,
    pub constraints: Vec<String>,
}

/// Search algorithm configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub algorithm: SearchAlgorithm,
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub evaluation_budget: usize,
}

/// Search algorithms
#[derive(Debug, Clone)]
pub enum SearchAlgorithm {
    Evolutionary,
    ReinforcementLearning,
    Differentiable,
    Random,
    Bayesian,
}

/// Parameter range for search space
#[derive(Debug, Clone)]
pub struct ParameterRange {
    pub min: f64,
    pub max: f64,
    pub log_scale: bool,
    pub discrete_values: Option<Vec<f64>>,
}

/// NAS Search Result
#[derive(Debug, Clone)]
pub struct NASSearchResult {
    pub best_architecture: Architecture,
    pub best_performance: f64,
    pub search_history: Vec<ArchitecturePerformance>,
    pub total_evaluations: usize,
    pub search_time: std::time::Duration,
    pub convergence_metrics: ConvergenceMetrics,
    pub experiment_summary: ExperimentSummary,
}

/// Architecture performance record
#[derive(Debug, Clone)]
pub struct ArchitecturePerformance {
    pub architecture: Architecture,
    pub performance: f64,
    pub evaluation_time: std::time::Duration,
    pub resource_usage: ResourceUsage,
    pub metadata: HashMap<String, String>,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub gpu_memory_mb: u64,
    pub cpu_time_seconds: f64,
    pub gpu_time_seconds: f64,
    pub peak_memory_mb: u64,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub final_improvement_rate: f64,
    pub diversity_score: f64,
    pub exploration_exploitation_ratio: f64,
    pub regret_bounds: Option<f64>,
}

/// Integrated NAS Research Framework
/// Provides seamless integration between NAS algorithms and research tracking
pub struct IntegratedNASFramework {
    /// Research framework instance
    research_framework: Arc<RwLock<UnifiedResearchFramework>>,
    /// NAS-specific experiment contexts
    experiment_contexts: HashMap<String, NASExperimentContext>,
    /// Performance prediction models
    performance_predictors: HashMap<String, Box<dyn PerformancePredictor>>,
    /// Search algorithm registry
    search_algorithms: HashMap<SearchAlgorithm, Box<dyn IntegratedSearchAlgorithm>>,
    /// Automated research pipelines
    automated_pipelines: Vec<AutomatedNASPiping>,
}

/// Performance predictor trait
#[derive(Debug)]
pub trait PerformancePredictor: Send + Sync {
    /// Predict performance for an architecture
    fn predict(&self, architecture: &Architecture, context: &NASExperimentContext) -> Result<f64>;

    /// Update predictor with new training data
    fn update(&mut self, architecture: &Architecture, actual_performance: f64) -> Result<()>;

    /// Get prediction confidence
    fn confidence(&self, architecture: &Architecture) -> f64;
}

/// Integrated search algorithm trait
pub trait IntegratedSearchAlgorithm: Send + Sync {
    /// Execute NAS search with research framework integration
    fn search_with_tracking(
        &self,
        context: &NASExperimentContext,
        framework: &mut UnifiedResearchFramework,
        evaluator: Arc<dyn ArchitectureEvaluator>,
        space: &ArchitectureSpace,
    ) -> Result<NASSearchResult>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Automated NAS Pipeline
#[derive(Debug)]
pub struct AutomatedNASPiping {
    pub name: String,
    pub description: String,
    pub pipeline_config: ResearchPipelineConfig,
    pub search_algorithms: Vec<SearchAlgorithm>,
    pub evaluation_strategies: Vec<EvaluationStrategy>,
    pub optimization_goals: Vec<OptimizationGoal>,
}

/// Research pipeline configuration
#[derive(Debug, Clone)]
pub struct ResearchPipelineConfig {
    pub max_runtime_seconds: u64,
    pub budget_constraints: BudgetConstraints,
    pub quality_thresholds: QualityThresholds,
    pub parallel_evaluations: usize,
    pub enable_early_stopping: bool,
}

/// Budget constraints
#[derive(Debug, Clone)]
pub struct BudgetConstraints {
    pub max_gpu_hours: f64,
    pub max_cpu_hours: f64,
    pub max_memory_gb: f64,
    pub max_evaluations: usize,
}

/// Quality thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_accuracy: f64,
    pub max_model_size_mb: f64,
    pub max_inference_time_ms: f64,
    pub convergence_tolerance: f64,
}

/// Evaluation strategies
#[derive(Debug, Clone)]
pub enum EvaluationStrategy {
    FullTraining,
    ProxyEvaluation,
    MultiFidelity,
    TransferLearning,
}

/// Optimization goals
#[derive(Debug, Clone)]
pub enum OptimizationGoal {
    MaximizeAccuracy,
    MinimizeModelSize,
    MinimizeLatency,
    MinimizeEnergyConsumption,
    MultiObjective(Vec<(String, f64)>),
}

impl IntegratedNASFramework {
    /// Create new integrated NAS framework
    pub fn new(research_framework: Arc<RwLock<UnifiedResearchFramework>>) -> Self {
        let mut framework = Self {
            research_framework,
            experiment_contexts: HashMap::new(),
            performance_predictors: HashMap::new(),
            search_algorithms: HashMap::new(),
            automated_pipelines: Vec::new(),
        };

        // Register default search algorithms
        framework.register_search_algorithm(SearchAlgorithm::Evolutionary, Box::new(EvolutionarySearchIntegration::new()));
        framework.register_search_algorithm(SearchAlgorithm::ReinforcementLearning, Box::new(RLSearchIntegration::new()));
        framework.register_search_algorithm(SearchAlgorithm::Differentiable, Box::new(DartsSearchIntegration::new()));

        framework
    }

    /// Start integrated NAS experiment
    pub fn start_nas_experiment(&mut self, context: NASExperimentContext) -> Result<String> {
        let experiment_id = format!("nas_{}_{}", context.domain, context.task);
        let experiment_name = format!("NAS Search: {} - {}", context.domain, context.task);
        let experiment_description = format!(
            "Neural Architecture Search for {} task on {} dataset with {} algorithm",
            context.task, context.dataset.name, format!("{:?}", context.search_config.algorithm)
        );

        // Create experiment in research framework
        let framework = self.research_framework.write().unwrap();
        let tracker = framework.create_experiment(
            experiment_id.clone(),
            experiment_name,
            experiment_description,
        );

        // Log NAS-specific metadata
        tracker.log_hyperparameter(
            "nas_algorithm".to_string(),
            format!("{:?}", context.search_config.algorithm).into(),
            Some("Neural architecture search algorithm used".to_string()),
        );
        tracker.log_hyperparameter(
            "search_space_size".to_string(),
            context.search_space_config.max_layers.into(),
            Some("Maximum number of layers in search space".to_string()),
        );
        tracker.log_hyperparameter(
            "population_size".to_string(),
            context.search_config.population_size.into(),
            Some("Population size for evolutionary algorithms".to_string()),
        );

        // Store context
        self.experiment_contexts.insert(experiment_id.clone(), context);

        Ok(experiment_id)
    }

    /// Execute integrated NAS search
    pub fn execute_nas_search(
        &mut self,
        experiment_id: &str,
        evaluator: Arc<dyn ArchitectureEvaluator>,
        space: &ArchitectureSpace,
    ) -> Result<NASSearchResult> {
        let context = self.experiment_contexts.get(experiment_id)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Experiment context not found for {}", experiment_id),
            })?
            .clone();

        let algorithm = self.search_algorithms.get(&context.search_config.algorithm)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Search algorithm {:?} not registered", context.search_config.algorithm),
            })?;

        let mut framework = self.research_framework.write().unwrap();

        // Execute search with research framework integration
        algorithm.search_with_tracking(&context, &mut framework, evaluator, space)
    }

    /// Start automated NAS pipeline
    pub fn start_automated_pipeline(
        &mut self,
        pipeline_name: &str,
        base_context: NASExperimentContext,
    ) -> Result<String> {
        let pipeline = self.automated_pipelines.iter()
            .find(|p| p.name == pipeline_name)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Automated pipeline {} not found", pipeline_name),
            })?;

        // Create pipeline experiment
        let pipeline_experiment_id = format!("auto_nas_{}_{}", pipeline_name, base_context.domain);
        let experiment_name = format!("Automated NAS: {}", pipeline_name);
        let experiment_description = format!(
            "Automated neural architecture search pipeline with multiple algorithms and strategies"
        );

        let framework = self.research_framework.write().unwrap();
        let tracker = framework.create_experiment(
            pipeline_experiment_id.clone(),
            experiment_name,
            experiment_description,
        );

        // Execute pipeline algorithms
        let mut results = Vec::new();
        for algorithm in &pipeline.search_algorithms {
            if let Some(search_alg) = self.search_algorithms.get(algorithm) {
                let mut context = base_context.clone();
                context.search_config.algorithm = algorithm.clone();
                context.experiment_id = format!("{}_{:?}", pipeline_experiment_id, algorithm);

                // Note: In real implementation, would need to handle concurrent execution
                // and resource management across multiple algorithms
            }
        }

        Ok(pipeline_experiment_id)
    }

    /// Register search algorithm
    pub fn register_search_algorithm(
        &mut self,
        algorithm_type: SearchAlgorithm,
        algorithm: Box<dyn IntegratedSearchAlgorithm>,
    ) {
        self.search_algorithms.insert(algorithm_type, algorithm);
    }

    /// Register performance predictor
    pub fn register_performance_predictor(
        &mut self,
        domain: String,
        predictor: Box<dyn PerformancePredictor>,
    ) {
        self.performance_predictors.insert(domain, predictor);
    }

    /// Add automated pipeline
    pub fn add_automated_pipeline(&mut self, pipeline: AutomatedNASPiping) {
        self.automated_pipelines.push(pipeline);
    }

    /// Get experiment summary with NAS metrics
    pub fn get_experiment_summary(&self, experiment_id: &str) -> Result<NASEperimentSummary> {
        let framework = self.research_framework.read().unwrap();
        let base_summary = framework.experiment_registry.get_experiment_summary(experiment_id)?;

        let context = self.experiment_contexts.get(experiment_id);

        Ok(NASEperimentSummary {
            base_summary,
            nas_context: context.cloned(),
            nas_metrics: NASMetrics {
                architectures_evaluated: 0, // Would be populated from search results
                best_architecture_score: None,
                search_efficiency: None,
                convergence_speed: None,
                diversity_maintained: None,
            },
        })
    }

    /// Generate NAS research report
    pub fn generate_nas_research_report(&self) -> Result<String> {
        let framework = self.research_framework.read().unwrap();

        let mut report = String::new();
        report.push_str("# Neural Architecture Search Research Report\n\n");

        // Summary statistics
        let total_experiments = self.experiment_contexts.len();
        report.push_str(&format!("## Summary\n"));
        report.push_str(&format!("- Total NAS Experiments: {}\n", total_experiments));
        report.push_str(&format!("- Active Experiments: {}\n", framework.health_status().experiments_active));
        report.push_str(&format!("- Registered Search Algorithms: {}\n", self.search_algorithms.len()));
        report.push_str(&format!("- Performance Predictors: {}\n", self.performance_predictors.len()));
        report.push_str(&format!("- Automated Pipelines: {}\n\n", self.automated_pipelines.len()));

        // Experiment details
        if !self.experiment_contexts.is_empty() {
            report.push_str("## NAS Experiments\n\n");
            for (id, context) in &self.experiment_contexts {
                report.push_str(&format!("### Experiment: {}\n", id));
                report.push_str(&format!("- Domain: {}\n", context.domain));
                report.push_str(&format!("- Task: {}\n", context.task));
                report.push_str(&format!("- Dataset: {}\n", context.dataset.name));
                report.push_str(&format!("- Algorithm: {:?}\n", context.search_config.algorithm));
                report.push_str(&format!("- Performance Prediction: {}\n", if context.performance_prediction { "Enabled" } else { "Disabled" }));
                report.push_str(&format!("- Joint NAS-HPO: {}\n\n", if context.joint_search { "Enabled" } else { "Disabled" }));
            }
        }

        Ok(report)
    }
}

/// NAS Experiment Summary
#[derive(Debug)]
pub struct NASEperimentSummary {
    pub base_summary: ExperimentSummary,
    pub nas_context: Option<NASExperimentContext>,
    pub nas_metrics: NASMetrics,
}

/// NAS-specific metrics
#[derive(Debug)]
pub struct NASMetrics {
    pub architectures_evaluated: usize,
    pub best_architecture_score: Option<f64>,
    pub search_efficiency: Option<f64>,
    pub convergence_speed: Option<f64>,
    pub diversity_maintained: Option<f64>,
}

/// Performance predictors implementations
pub mod predictors {
    use super::*;

    /// Linear regression performance predictor
    pub struct LinearRegressionPredictor {
        coefficients: Vec<f64>,
        intercept: f64,
        trained: bool,
    }

    impl LinearRegressionPredictor {
        pub fn new() -> Self {
            Self {
                coefficients: Vec::new(),
                intercept: 0.0,
                trained: false,
            }
        }
    }

    impl PerformancePredictor for LinearRegressionPredictor {
        fn predict(&self, architecture: &Architecture, _context: &NASExperimentContext) -> Result<f64> {
            if !self.trained {
                return Err(NNError::NotInitialized {
                    message: "Performance predictor not trained".to_string(),
                });
            }

            // Simple prediction based on number of parameters and layers
            let num_params = architecture.num_parameters() as f64;
            let num_layers = architecture.layers.len() as f64;

            let mut prediction = self.intercept;
            if !self.coefficients.is_empty() {
                prediction += self.coefficients[0] * num_params;
            }
            if self.coefficients.len() > 1 {
                prediction += self.coefficients[1] * num_layers;
            }

            Ok(prediction.max(0.0).min(1.0)) // Clamp to [0, 1]
        }

        fn update(&mut self, architecture: &Architecture, actual_performance: f64) -> Result<()> {
            // Simple online learning update (stochastic gradient descent)
            let num_params = architecture.num_parameters() as f64;
            let num_layers = architecture.layers.len() as f64;

            let prediction = self.predict(architecture, &NASExperimentContext {
                experiment_id: "".to_string(),
                domain: "".to_string(),
                task: "".to_string(),
                dataset: DatasetInfo {
                    name: "".to_string(),
                    size: 0,
                    input_shape: vec![],
                    output_classes: 0,
                    metadata: HashMap::new(),
                },
                search_space_config: SearchSpaceConfig {
                    max_layers: 0,
                    available_operations: vec![],
                    parameter_ranges: HashMap::new(),
                    constraints: vec![],
                },
                search_config: SearchConfig {
                    algorithm: SearchAlgorithm::Random,
                    population_size: 0,
                    generations: 0,
                    mutation_rate: 0.0,
                    crossover_rate: 0.0,
                    evaluation_budget: 0,
                },
                performance_prediction: false,
                joint_search: false,
            }).unwrap_or(0.5);

            let error = actual_performance - prediction;

            if self.coefficients.is_empty() {
                self.coefficients = vec![0.0, 0.0];
            }

            // Update coefficients with learning rate
            let learning_rate = 0.01;
            self.coefficients[0] += learning_rate * error * num_params;
            self.coefficients[1] += learning_rate * error * num_layers;
            self.intercept += learning_rate * error;

            self.trained = true;

            Ok(())
        }

        fn confidence(&self, _architecture: &Architecture) -> f64 {
            if !self.trained {
                0.0
            } else {
                0.8 // Placeholder confidence score
            }
        }
    }
}

/// Search algorithm integrations
pub mod search_integrations {
    use super::*;

    /// Evolutionary NAS integration
    pub struct EvolutionarySearchIntegration {
        base_algorithm: EvolutionaryNAS,
    }

    impl EvolutionarySearchIntegration {
        pub fn new() -> Self {
            Self {
                base_algorithm: EvolutionaryNAS::new(50, 0.1, 0.8), // Default parameters
            }
        }
    }

    impl IntegratedSearchAlgorithm for EvolutionarySearchIntegration {
        fn search_with_tracking(
            &self,
            context: &NASExperimentContext,
            framework: &mut UnifiedResearchFramework,
            evaluator: Arc<dyn ArchitectureEvaluator>,
            space: &ArchitectureSpace,
        ) -> Result<NASSearchResult> {
            let start_time = Instant::now();

            let tracker = framework.create_experiment(
                format!("{}_evo", context.experiment_id),
                "Evolutionary NAS Search".to_string(),
                "Evolutionary algorithm based neural architecture search".to_string(),
            );

            let mut search_history = Vec::new();

            // Initialize population
            let mut population = Vec::new();
            for _ in 0..context.search_config.population_size {
                let arch = space.sample_random(context.search_space_config.max_layers)?;
                population.push(arch);
            }

            // Evolutionary search loop
            for generation in 0..context.search_config.generations {
                // Evaluate current population
                let mut evaluated_population = Vec::new();

                for architecture in &population {
                    let eval_start = Instant::now();
                    let result = evaluator.evaluate(architecture)?;
                    let eval_time = eval_start.elapsed();

                    // Record in research framework
                    tracker.record_metric(
                        "architecture_accuracy".to_string(),
                        result.accuracy,
                        Some(HashMap::from([
                            ("generation".to_string(), generation.to_string()),
                            ("architecture_id".to_string(), format!("arch_{}", evaluated_population.len())),
                        ])),
                    );

                    let performance = ArchitecturePerformance {
                        architecture: architecture.clone(),
                        performance: result.accuracy,
                        evaluation_time: eval_time,
                        resource_usage: ResourceUsage {
                            gpu_memory_mb: 1024, // Placeholder
                            cpu_time_seconds: eval_time.as_secs_f64(),
                            gpu_time_seconds: eval_time.as_secs_f64(),
                            peak_memory_mb: 2048, // Placeholder
                        },
                        metadata: HashMap::from([
                            ("generation".to_string(), generation.to_string()),
                            ("fitness".to_string(), result.accuracy.to_string()),
                        ]),
                    };

                    evaluated_population.push(performance.clone());
                    search_history.push(performance);
                }

                // Sort by performance
                evaluated_population.sort_by(|a, b| b.performance.partial_cmp(&a.performance).unwrap());

                // Evolve to next generation
                let mut next_population = Vec::new();

                // Elitism - keep best performers
                let elite_count = (context.search_config.population_size as f64 * 0.1) as usize;
                for i in 0..elite_count.min(evaluated_population.len()) {
                    next_population.push(evaluated_population[i].architecture.clone());
                }

                // Generate offspring through crossover and mutation
                while next_population.len() < context.search_config.population_size {
                    // Tournament selection
                    let parent1 = self.tournament_selection(&evaluated_population, 3);
                    let parent2 = self.tournament_selection(&evaluated_population, 3);

                    // Crossover
                    if rand::random::<f64>() < context.search_config.crossover_rate {
                        let offspring = self.crossover(&parent1.architecture, &parent2.architecture, space)?;
                        next_population.push(offspring);
                    } else {
                        next_population.push(parent1.architecture.clone());
                    }

                    // Mutation
                    if rand::random::<f64>() < context.search_config.mutation_rate && !next_population.is_empty() {
                        let idx = next_population.len() - 1;
                        next_population[idx] = self.mutate(&next_population[idx], space)?;
                    }
                }

                population = next_population;

                // Log generation metrics
                let best_fitness = evaluated_population.first()
                    .map(|p| p.performance)
                    .unwrap_or(0.0);

                tracker.record_metric(
                    "generation_best_fitness".to_string(),
                    best_fitness,
                    Some(HashMap::from([("generation".to_string(), generation.to_string())])),
                );
            }

            let total_time = start_time.elapsed();
            let best_performance = search_history.iter()
                .map(|p| p.performance)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            let best_architecture = search_history.iter()
                .max_by(|a, b| a.performance.partial_cmp(&b.performance).unwrap())
                .map(|p| p.architecture.clone())
                .unwrap_or_else(|| population[0].clone());

            Ok(NASSearchResult {
                best_architecture,
                best_performance,
                search_history,
                total_evaluations: search_history.len(),
                search_time: total_time,
                convergence_metrics: ConvergenceMetrics {
                    final_improvement_rate: 0.05, // Placeholder
                    diversity_score: 0.8, // Placeholder
                    exploration_exploitation_ratio: 0.6, // Placeholder
                    regret_bounds: Some(0.1), // Placeholder
                },
                experiment_summary: tracker.summarize(),
            })
        }

        fn name(&self) -> &str {
            "Evolutionary NAS Integration"
        }
    }

    impl EvolutionarySearchIntegration {
        fn tournament_selection<'a>(&self, population: &'a [ArchitecturePerformance], tournament_size: usize) -> &'a ArchitecturePerformance {
            let mut best = None;
            for _ in 0..tournament_size {
                let idx = rand::random::<usize>() % population.len();
                if let Some(current_best) = best {
                    if population[idx].performance > current_best.performance {
                        best = Some(&population[idx]);
                    }
                } else {
                    best = Some(&population[idx]);
                }
            }
            best.unwrap()
        }

        fn crossover(&self, parent1: &Architecture, parent2: &Architecture, space: &ArchitectureSpace) -> Result<Architecture> {
            // Simple single-point crossover
            let mut child = parent1.clone();

            let crossover_point = rand::random::<usize>() % parent1.layers.len().min(parent2.layers.len());

            for i in crossover_point..child.layers.len().min(parent2.layers.len()) {
                child.layers[i] = parent2.layers[i].clone();
            }

            child.validate()?;
            Ok(child)
        }

        fn mutate(&self, architecture: &Architecture, space: &ArchitectureSpace) -> Result<Architecture> {
            let mut mutated = architecture.clone();

            // Random layer mutation
            if !mutated.layers.is_empty() {
                let layer_idx = rand::random::<usize>() % mutated.layers.len();

                // Replace with random layer from search space
                let random_layer = space.sample_layer(&crate::nn::nas::search_space::LayerType::Conv2D)?;
                mutated.layers[layer_idx] = random_layer;
            }

            mutated.validate()?;
            Ok(mutated)
        }
    }

    /// Reinforcement Learning NAS integration
    pub struct RLSearchIntegration {
        base_algorithm: ReinforcementNAS,
    }

    impl RLSearchIntegration {
        pub fn new() -> Self {
            Self {
                base_algorithm: ReinforcementNAS::new(0.9, 0.95, 0.1, 1.0), // Default parameters
            }
        }
    }

    impl IntegratedSearchAlgorithm for RLSearchIntegration {
        fn search_with_tracking(
            &self,
            context: &NASExperimentContext,
            framework: &mut UnifiedResearchFramework,
            evaluator: Arc<dyn ArchitectureEvaluator>,
            space: &ArchitectureSpace,
        ) -> Result<NASSearchResult> {
            // Similar implementation to evolutionary but using RL
            // Placeholder for now
            Err(NNError::NotImplemented {
                operation: "RL NAS integration search".to_string(),
            })
        }

        fn name(&self) -> &str {
            "Reinforcement Learning NAS Integration"
        }
    }

    /// DARTS NAS integration
    pub struct DartsSearchIntegration {
        base_algorithm: DartsNAS,
    }

    impl DartsSearchIntegration {
        pub fn new() -> Self {
            Self {
                base_algorithm: DartsNAS::new(0.001, 0.0003, 0.2), // Default parameters
            }
        }
    }

    impl IntegratedSearchAlgorithm for DartsSearchIntegration {
        fn search_with_tracking(
            &self,
            context: &NASExperimentContext,
            framework: &mut UnifiedResearchFramework,
            evaluator: Arc<dyn ArchitectureEvaluator>,
            space: &ArchitectureSpace,
        ) -> Result<NASSearchResult> {
            // Placeholder for DARTS implementation
            Err(NNError::NotImplemented {
                operation: "DARTS NAS integration search".to_string(),
            })
        }

        fn name(&self) -> &str {
            "DARTS NAS Integration"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::nas::search_space::{ArchitectureSpace, ArchitectureType};

    #[test]
    fn test_integrated_nas_framework_creation() {
        let research_framework = Arc::new(RwLock::new(UnifiedResearchFramework::new()));
        let nas_framework = IntegratedNASFramework::new(research_framework);
        assert!(!nas_framework.search_algorithms.is_empty());
    }

    #[test]
    fn test_nas_experiment_context() {
        let context = NASExperimentContext {
            experiment_id: "test_exp".to_string(),
            domain: "computer_vision".to_string(),
            task: "classification".to_string(),
            dataset: DatasetInfo {
                name: "cifar10".to_string(),
                size: 50000,
                input_shape: vec![32, 32, 3],
                output_classes: 10,
                metadata: HashMap::new(),
            },
            search_space_config: SearchSpaceConfig {
                max_layers: 10,
                available_operations: vec!["conv2d".to_string(), "linear".to_string()],
                parameter_ranges: HashMap::new(),
                constraints: vec![],
            },
            search_config: SearchConfig {
                algorithm: SearchAlgorithm::Evolutionary,
                population_size: 20,
                generations: 5,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
                evaluation_budget: 100,
            },
            performance_prediction: true,
            joint_search: false,
        };

        assert_eq!(context.domain, "computer_vision");
        assert_eq!(context.task, "classification");
    }

    #[test]
    fn test_performance_predictor() {
        use predictors::LinearRegressionPredictor;

        let mut predictor = LinearRegressionPredictor::new();
        let space = ArchitectureSpace::new(ArchitectureType::CNN);

        // Test untrained predictor
        let architecture = space.sample_random(5).unwrap();
        let context = NASExperimentContext {
            experiment_id: "test".to_string(),
            domain: "cv".to_string(),
            task: "cls".to_string(),
            dataset: DatasetInfo {
                name: "test".to_string(),
                size: 1000,
                input_shape: vec![28, 28, 1],
                output_classes: 10,
                metadata: HashMap::new(),
            },
            search_space_config: SearchSpaceConfig {
                max_layers: 5,
                available_operations: vec![],
                parameter_ranges: HashMap::new(),
                constraints: vec![],
            },
            search_config: SearchConfig {
                algorithm: SearchAlgorithm::Random,
                population_size: 10,
                generations: 5,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
                evaluation_budget: 50,
            },
            performance_prediction: false,
            joint_search: false,
        };

        // Should fail when untrained
        assert!(predictor.predict(&architecture, &context).is_err());

        // Train predictor
        predictor.update(&architecture, 0.85).unwrap();

        // Should work when trained
        let prediction = predictor.predict(&architecture, &context).unwrap();
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }
}
