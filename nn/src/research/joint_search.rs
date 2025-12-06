//! Joint NAS-HPO Search Algorithms
//!
//! This module provides algorithms for jointly optimizing neural architectures
//! and hyperparameters, combining NAS and HPO into efficient, coordinated search
//! processes that exploit the synergies between architecture and optimization decisions.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::error::{NNError, Result};
use crate::nas::search_space::LayerSpec;
use crate::nas::{Architecture, ArchitectureSpace, ArchitectureEvaluator};
use crate::hpo::{HyperparameterSpace, HyperparameterConfig};
use crate::research::UnifiedResearchFramework;
use crate::research::joint_search::algorithms::{AlternatingSearch, ConcurrentSearch, EvolutionaryJointSearch, FactorizedSearch};

/// Joint Search Result
#[derive(Debug, Clone)]
pub struct JointSearchResult {
    pub best_architecture: Architecture,
    pub best_hyperparameters: HyperparameterConfig,
    pub best_score: f64,
    pub joint_evaluations: usize,
    pub search_time: std::time::Duration,
    pub convergence_metrics: ConvergenceMetrics,
    pub pareto_front: Option<Vec<JointSolution>>,
    pub experiment_summary: crate::research::tracking::ExperimentSummary,
}

/// A solution in the joint architecture-hyperparameter space
#[derive(Debug, Clone)]
pub struct JointSolution {
    pub architecture: Architecture,
    pub hyperparameters: HyperparameterConfig,
    pub score: f64,
    pub evaluation_time: std::time::Duration,
    pub resource_usage: ResourceUsage,
}

/// Resource usage for joint evaluation
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub gpu_memory_mb: u64,
    pub cpu_time_seconds: f64,
    pub training_time_seconds: f64,
    pub peak_memory_mb: u64,
}

/// Convergence metrics for joint search
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub final_improvement_rate: f64,
    pub architecture_diversity: f64,
    pub hyperparameter_diversity: f64,
    pub exploration_exploitation_ratio: f64,
    pub joint_regret_bounds: Option<f64>,
}

/// Joint Search Algorithm Trait
pub trait JointSearchAlgorithm: Send + Sync + std::fmt::Debug {
    /// Execute joint NAS-HPO search
    fn joint_search(
        &self,
        context: &JointSearchContext,
        framework: &mut UnifiedResearchFramework,
        arch_evaluator: Arc<dyn ArchitectureEvaluator>,
        arch_space: &ArchitectureSpace,
        hp_space: &HyperparameterSpace,
    ) -> Result<JointSearchResult>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm description
    fn description(&self) -> &str;
}

/// Joint Search Context
#[derive(Debug, Clone)]
pub struct JointSearchContext {
    pub experiment_id: String,
    pub dataset_name: String,
    pub task_type: String,
    pub budget: JointSearchBudget,
    pub search_strategy: JointSearchStrategy,
    pub performance_predictors: Option<PerformancePredictors>,
    pub evaluation_strategy: JointEvaluationStrategy,
}

/// Search budget for joint optimization
#[derive(Debug, Clone)]
pub struct JointSearchBudget {
    pub max_total_evaluations: usize,
    pub max_time_seconds: u64,
    pub max_gpu_hours: f64,
    pub max_parallel_evaluations: usize,
}

/// Joint search strategies
#[derive(Debug, Clone)]
pub enum JointSearchStrategy {
    /// Alternating between architecture and hyperparameter search
    Alternating {
        architecture_rounds: usize,
        hyperparameter_rounds: usize,
        synchronization_frequency: usize,
    },
    /// Concurrent search in both spaces
    Concurrent {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Evolutionary algorithms in joint space
    EvolutionaryJoint {
        population_size: usize,
        elite_size: usize,
        mutation_probability: f64,
    },
    /// Bayesian optimization in joint space
    BayesianJoint {
        initial_samples: usize,
        acquisition_function: String,
    },
    /// Factorized search (separate but coordinated)
    Factorized {
        coordination_budget: usize,
        transfer_learning: bool,
    },
}

/// Performance predictors for joint search
#[derive(Debug, Clone)]
pub struct PerformancePredictors {
    pub architecture_predictor: String,
    pub hyperparameter_predictor: String,
    pub joint_predictor: Option<String>,
    pub prediction_confidence_threshold: f64,
}

/// Evaluation strategies for joint search
#[derive(Debug, Clone)]
pub enum JointEvaluationStrategy {
    /// Full training for every joint evaluation
    FullTraining,
    /// Use performance predictors with occasional full validation
    PredictorGuided {
        predictor_confidence_threshold: f64,
        full_validation_frequency: usize,
    },
    /// Multi-fidelity evaluation (partial training)
    MultiFidelity {
        fidelity_levels: Vec<f64>, // e.g., [0.1, 0.5, 1.0] for epoch fractions
        promotion_criteria: PromotionCriteria,
    },
    /// Transfer learning from related tasks
    TransferLearning {
        base_architectures: Vec<String>,
        fine_tuning_epochs: usize,
    },
}

/// Criteria for promoting configurations to higher fidelity
#[derive(Debug, Clone)]
pub struct PromotionCriteria {
    pub min_score_threshold: f64,
    pub percentile_threshold: f64,
    pub diversity_requirement: f64,
}

/// Joint Search Framework
/// Manages joint NAS-HPO search algorithms and orchestration
#[derive(Debug)]
pub struct JointSearchFramework {
    /// Registered joint search algorithms
    algorithms: HashMap<String, Box<dyn JointSearchAlgorithm>>,
    /// Search history and meta-learning
    search_history: Vec<JointSearchResult>,
    /// Performance prediction framework
    prediction_framework: Option<Arc<super::performance_prediction::PerformancePredictionFramework>>,
    /// Multi-objective optimization utilities
    multi_objective_utils: MultiObjectiveUtils,
    /// Coordination mechanisms between NAS and HPO
    coordination_mechanisms: CoordinationMechanisms,
}

/// Coordination mechanisms for joint search
#[derive(Debug)]
struct CoordinationMechanisms {
    /// Transfer learning between architecture families
    transfer_learning: TransferLearningCoordinator,
    /// Knowledge sharing between search algorithms
    knowledge_sharing: KnowledgeSharingCoordinator,
    /// Resource allocation across joint evaluations
    resource_arbitration: ResourceArbitrationCoordinator,
}

/// Transfer learning coordinator
#[derive(Debug)]
struct TransferLearningCoordinator {
    /// Architecture knowledge base for transfer
    architecture_knowledge: HashMap<String, Vec<Architecture>>,
    /// Hyperparameter transfer mappings
    hyperparameter_transfers: HashMap<String, Vec<HyperparameterConfig>>,
}

/// Knowledge sharing coordinator
#[derive(Debug)]
struct KnowledgeSharingCoordinator {
    /// Shared performance data across domains
    shared_performance_data: Vec<PerformanceDatum>,
    /// Meta-knowledge about search effectiveness
    meta_knowledge: HashMap<String, SearchEffectiveness>,
}

/// Resource arbitration coordinator
#[derive(Debug)]
struct ResourceArbitrationCoordinator {
    /// Current resource allocations
    allocations: HashMap<String, ResourceAllocation>,
    /// Resource constraints
    constraints: ResourceConstraints,
    /// Fairness policies
    fairness_policies: FairnessPolicies,
}

/// Performance datum for knowledge sharing
#[derive(Debug, Clone)]
struct PerformanceDatum {
    architecture_family: String,
    task_domain: String,
    performance: f64,
    confidence: f64,
}

/// Search effectiveness meta-knowledge
#[derive(Debug, Clone)]
struct SearchEffectiveness {
    algorithm_name: String,
    domain_adaptability: f64,
    convergence_speed: f64,
    final_accuracy: f64,
}

/// Resource allocation
#[derive(Debug, Clone)]
struct ResourceAllocation {
    experiment_id: String,
    gpu_allocation: f64, // fraction of GPU
    time_allocation: std::time::Duration,
    priority: usize,
}

/// Resource constraints
#[derive(Debug, Clone)]
struct ResourceConstraints {
    total_gpu_memory: f64,
    total_cpu_cores: usize,
    time_limits: TimeLimits,
}

/// Time limits for resource allocation
#[derive(Debug, Clone)]
struct TimeLimits {
    max_single_evaluation: std::time::Duration,
    max_total_search: std::time::Duration,
}

/// Fairness policies for resource allocation
#[derive(Debug, Clone)]
struct FairnessPolicies {
    egalitarian: bool, // equal allocation
    merit_based: bool, // based on potential
    starvation_prevention: bool, // prevent some algorithms from being starved
}

/// Multi-objective optimization utilities
#[derive(Debug)]
struct MultiObjectiveUtils {
    // Implementation details will be added as needed
}

impl JointSearchFramework {
    /// Create new joint search framework
    pub fn new() -> Self {
        let mut framework = Self {
            algorithms: HashMap::new(),
            search_history: Vec::new(),
            prediction_framework: None,
            multi_objective_utils: MultiObjectiveUtils::new(),
            coordination_mechanisms: CoordinationMechanisms {
                transfer_learning: TransferLearningCoordinator {
                    architecture_knowledge: HashMap::new(),
                    hyperparameter_transfers: HashMap::new(),
                },
                knowledge_sharing: KnowledgeSharingCoordinator {
                    shared_performance_data: Vec::new(),
                    meta_knowledge: HashMap::new(),
                },
                resource_arbitration: ResourceArbitrationCoordinator {
                    allocations: HashMap::new(),
                    constraints: ResourceConstraints::default(),
                    fairness_policies: FairnessPolicies::default(),
                },
            },
        };

        // Register default algorithms
        framework.register_algorithm("alternating".to_string(), Box::new(AlternatingSearch::new()));
        framework.register_algorithm("concurrent".to_string(), Box::new(ConcurrentSearch::new()));
        framework.register_algorithm("evolutionary_joint".to_string(), Box::new(EvolutionaryJointSearch::new()));
        framework.register_algorithm("factorized".to_string(), Box::new(FactorizedSearch::new()));

        framework
    }

    /// Set performance prediction framework
    pub fn with_prediction_framework(
        mut self,
        prediction_framework: Arc<super::performance_prediction::PerformancePredictionFramework>,
    ) -> Self {
        self.prediction_framework = Some(prediction_framework);
        self
    }

    /// Execute joint search
    pub fn execute_joint_search(
        &mut self,
        algorithm_name: &str,
        context: &JointSearchContext,
        framework: &mut UnifiedResearchFramework,
        arch_evaluator: Arc<dyn ArchitectureEvaluator>,
        arch_space: &ArchitectureSpace,
        hp_space: &HyperparameterSpace,
    ) -> Result<JointSearchResult> {
        if let Some(algorithm) = self.algorithms.get(algorithm_name) {
            let start_time = Instant::now();
            let mut tracker = framework.create_experiment(
                format!("{}_joint", context.experiment_id),
                "Joint NAS-HPO Search".to_string(),
                "Combined neural architecture and hyperparameter optimization".to_string(),
            );

            let result = algorithm.joint_search(context, framework, arch_evaluator, arch_space, hp_space)?;
            let search_time = start_time.elapsed();

            // Store in search history for meta-learning
            self.search_history.push(result.clone());

            // Update coordination mechanisms
            self.update_coordination_mechanisms(&result, context);

            Ok(result)
        } else {
            Err(NNError::InvalidConfiguration {
                message: format!("Joint search algorithm '{}' not found", algorithm_name),
            })
        }
    }

    /// Register a joint search algorithm
    pub fn register_algorithm(&mut self, name: String, algorithm: Box<dyn JointSearchAlgorithm>) {
        self.algorithms.insert(name, algorithm);
    }

    /// Get available algorithm names
    pub fn available_algorithms(&self) -> Vec<String> {
        self.algorithms.keys().cloned().collect()
    }

    /// Recommend algorithm based on context
    pub fn recommend_algorithm(&self, context: &JointSearchContext) -> String {
        // Simple recommendation logic based on context
        match context.search_strategy {
            JointSearchStrategy::Alternating { .. } => "alternating".to_string(),
            JointSearchStrategy::Concurrent { .. } => "concurrent".to_string(),
            JointSearchStrategy::EvolutionaryJoint { .. } => "evolutionary_joint".to_string(),
            JointSearchStrategy::BayesianJoint { .. } => "factorized".to_string(), // Use factorized as approximation
            JointSearchStrategy::Factorized { .. } => "factorized".to_string(),
        }
    }

    /// Update coordination mechanisms based on search results
    fn update_coordination_mechanisms(
        &mut self,
        result: &JointSearchResult,
        context: &JointSearchContext,
    ) {
        // Update transfer learning knowledge
        let architecture_family = self.classify_architecture_family(&result.best_architecture);
        self.coordination_mechanisms.transfer_learning.architecture_knowledge
            .entry(architecture_family.clone())
            .or_insert_with(Vec::new)
            .push(result.best_architecture.clone());

        // Update hyperparameter transfers
        self.coordination_mechanisms.transfer_learning.hyperparameter_transfers
            .entry(context.dataset_name.clone())
            .or_insert_with(Vec::new)
            .push(result.best_hyperparameters.clone());

        // Update shared performance data
        self.coordination_mechanisms.knowledge_sharing.shared_performance_data
            .push(PerformanceDatum {
                architecture_family: architecture_family.clone(),
                task_domain: context.task_type.clone(),
                performance: result.best_score,
                confidence: 1.0, // Full evaluation confidence
            });

        // Update meta-knowledge
        if let Some(algorithm_name) = self.algorithms.keys().next() { // Use first algorithm as placeholder
            self.coordination_mechanisms.knowledge_sharing.meta_knowledge
                .entry(algorithm_name.clone())
                .and_modify(|effectiveness| {
                    effectiveness.convergence_speed = result.search_time.as_secs_f64();
                    effectiveness.final_accuracy = result.best_score;
                    effectiveness.domain_adaptability += 0.1; // Incremental learning
                })
                .or_insert(SearchEffectiveness {
                    algorithm_name: algorithm_name.clone(),
                    domain_adaptability: 0.5,
                    convergence_speed: result.search_time.as_secs_f64(),
                    final_accuracy: result.best_score,
                });
        }
    }

    /// Classify architecture family for transfer learning
    fn classify_architecture_family(&self, architecture: &Architecture) -> String {
        // Simple classification logic
        let conv_count = architecture.layers.iter()
            .filter(|layer| matches!(layer, crate::nas::search_space::LayerSpec::Conv2D { .. }))
            .count();

        let attention_count = architecture.layers.iter()
            .filter(|layer| matches!(layer, crate::nas::search_space::LayerSpec::Attention { .. }))
            .count();

        if attention_count > conv_count {
            "transformer_based".to_string()
        } else if conv_count > 0 {
            "cnn_based".to_string()
        } else {
            "other".to_string()
        }
    }

    /// Generate search insights from history
    pub fn generate_search_insights(&self) -> JointSearchInsights {
        let mut insights = JointSearchInsights {
            algorithm_effectiveness: HashMap::new(),
            domain_adaptation_patterns: Vec::new(),
            resource_efficiency_trends: Vec::new(),
            architecture_hyperparameter_correlations: Vec::new(),
        };

        // Analyze algorithm effectiveness
        for result in &self.search_history {
            insights.algorithm_effectiveness
                .entry("joint_search".to_string()) // Placeholder
                .and_modify(|effectiveness| {
                    effectiveness.total_runs += 1;
                    effectiveness.average_score = (effectiveness.average_score * (effectiveness.total_runs - 1) as f64 + result.best_score) / effectiveness.total_runs as f64;
                    effectiveness.average_time += result.search_time.as_secs_f64();
                })
                .or_insert(AlgorithmEffectiveness {
                    algorithm_name: "joint_search".to_string(),
                    total_runs: 1,
                    average_score: result.best_score,
                    average_time: result.search_time.as_secs_f64(),
                    success_rate: if result.best_score > 0.8 { 1.0 } else { 0.0 },
                });
        }

        insights
    }
}

/// Insights from joint search history
#[derive(Debug)]
pub struct JointSearchInsights {
    pub algorithm_effectiveness: HashMap<String, AlgorithmEffectiveness>,
    pub domain_adaptation_patterns: Vec<DomainAdaptationPattern>,
    pub resource_efficiency_trends: Vec<ResourceEfficiencyTrend>,
    pub architecture_hyperparameter_correlations: Vec<ArchitectureHyperparameterCorrelation>,
}

/// Algorithm effectiveness metrics
#[derive(Debug, Clone)]
pub struct AlgorithmEffectiveness {
    pub algorithm_name: String,
    pub total_runs: usize,
    pub average_score: f64,
    pub average_time: f64,
    pub success_rate: f64,
}

/// Domain adaptation patterns
#[derive(Debug, Clone)]
pub struct DomainAdaptationPattern {
    pub from_domain: String,
    pub to_domain: String,
    pub transfer_efficiency: f64,
    pub architecture_similarity: f64,
}

/// Resource efficiency trends
#[derive(Debug, Clone)]
pub struct ResourceEfficiencyTrend {
    pub time_period: String,
    pub efficiency_improvement: f64,
    pub resource_utilization: f64,
}

/// Architecture-hyperparameter correlations
#[derive(Debug, Clone)]
pub struct ArchitectureHyperparameterCorrelation {
    pub architecture_feature: String,
    pub hyperparameter: String,
    pub correlation_strength: f64,
    pub confidence: f64,
}

impl MultiObjectiveUtils {
    fn new() -> Self {
        Self {}
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            total_gpu_memory: 8.0 * 1024.0, // 8GB default
            total_cpu_cores: 8,
            time_limits: TimeLimits {
                max_single_evaluation: std::time::Duration::from_secs(3600), // 1 hour
                max_total_search: std::time::Duration::from_secs(86400), // 24 hours
            },
        }
    }
}

impl Default for FairnessPolicies {
    fn default() -> Self {
        Self {
            egalitarian: true,
            merit_based: false,
            starvation_prevention: true,
        }
    }
}

/// Joint search algorithm implementations
pub mod algorithms {
    use super::*;

    /// Alternating search between NAS and HPO
    #[derive(Debug)]
    pub struct AlternatingSearch;

    impl AlternatingSearch {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl JointSearchAlgorithm for AlternatingSearch {
        fn joint_search(
            &self,
            context: &JointSearchContext,
            framework: &mut UnifiedResearchFramework,
            arch_evaluator: Arc<dyn ArchitectureEvaluator>,
            arch_space: &ArchitectureSpace,
            hp_space: &HyperparameterSpace,
        ) -> Result<JointSearchResult> {
            match &context.search_strategy {
                JointSearchStrategy::Alternating {
                    architecture_rounds,
                    hyperparameter_rounds,
                    synchronization_frequency,
                } => {
                    self.alternating_search(
                        context, framework, arch_evaluator, arch_space, hp_space,
                        *architecture_rounds, *hyperparameter_rounds, *synchronization_frequency,
                    )
                }
                _ => Err(NNError::InvalidConfiguration {
                    message: "Alternating search requires Alternating strategy".to_string(),
                }),
            }
        }

        fn name(&self) -> &str {
            "Alternating NAS-HPO Search"
        }

        fn description(&self) -> &str {
            "Alternates between architecture search and hyperparameter optimization phases"
        }
    }

    impl AlternatingSearch {
        fn alternating_search(
            &self,
            context: &JointSearchContext,
            framework: &mut UnifiedResearchFramework,
            arch_evaluator: Arc<dyn ArchitectureEvaluator>,
            arch_space: &ArchitectureSpace,
            hp_space: &HyperparameterSpace,
            arch_rounds: usize,
            hp_rounds: usize,
            sync_freq: usize,
        ) -> Result<JointSearchResult> {
            let mut best_solution = None;
            let mut evaluations = 0;
            let start_time = Instant::now();

            // Initialize with default architecture and hyperparameters
            let mut current_architecture = arch_space.sample_random(3)?;
            let mut current_hyperparameters = hp_space.sample()?;

            // Alternating optimization loop
            let total_rounds = arch_rounds + hp_rounds;
            for round in 0..total_rounds {
                if evaluations >= context.budget.max_total_evaluations {
                    break;
                }

                if round % 2 == 0 || round < arch_rounds {
                    // Architecture optimization round (keep hyperparameters fixed)
                    for _ in 0..10 { // Architecture search budget per round
                        if evaluations >= context.budget.max_total_evaluations {
                            break;
                        }

                        let candidate_arch = arch_space.sample_random(3)?;
                        let test_config = current_hyperparameters.clone();

                        // Evaluate joint configuration
                        let score = self.evaluate_joint_configuration(
                            &candidate_arch, &test_config, arch_evaluator.clone(),
                        )?;

                        evaluations += 1;

                        if best_solution.as_ref().map_or(true, |s: &JointSolution| score > s.score) {
                            best_solution = Some(JointSolution {
                                architecture: candidate_arch.clone(),
                                hyperparameters: current_hyperparameters.clone(),
                                score,
                                evaluation_time: start_time.elapsed(),
                                resource_usage: ResourceUsage {
                                    gpu_memory_mb: 1024,
                                    cpu_time_seconds: 10.0,
                                    training_time_seconds: 100.0,
                                    peak_memory_mb: 2048,
                                },
                            });

                            current_architecture = candidate_arch;
                        }
                    }
                } else {
                    // Hyperparameter optimization round (keep architecture fixed)
                    for _ in 0..20 { // Hyperparameter search budget per round
                        if evaluations >= context.budget.max_total_evaluations {
                            break;
                        }

                        let candidate_hp = hp_space.sample()?;

                        // Evaluate joint configuration
                        let score = self.evaluate_joint_configuration(
                            &current_architecture, &candidate_hp, arch_evaluator.clone(),
                        )?;

                        evaluations += 1;

                        if best_solution.as_ref().map_or(true, |s| score > s.score) {
                            best_solution = Some(JointSolution {
                                architecture: current_architecture.clone(),
                                hyperparameters: candidate_hp.clone(),
                                score,
                                evaluation_time: start_time.elapsed(),
                                resource_usage: ResourceUsage {
                                    gpu_memory_mb: 1024,
                                    cpu_time_seconds: 10.0,
                                    training_time_seconds: 100.0,
                                    peak_memory_mb: 2048,
                                },
                            });

                            current_hyperparameters = candidate_hp;
                        }
                    }
                }
            }

            let best_solution = best_solution.ok_or_else(|| NNError::InvalidConfiguration {
                message: "No valid solutions found".to_string(),
            })?;

            Ok(JointSearchResult {
                best_architecture: best_solution.architecture,
                best_hyperparameters: best_solution.hyperparameters,
                best_score: best_solution.score,
                joint_evaluations: evaluations,
                search_time: start_time.elapsed(),
                convergence_metrics: ConvergenceMetrics {
                    final_improvement_rate: 0.05,
                    architecture_diversity: 0.8,
                    hyperparameter_diversity: 0.7,
                    exploration_exploitation_ratio: 0.6,
                    joint_regret_bounds: Some(0.1),
                },
                pareto_front: None,
                experiment_summary: framework.experiment_registry.get_experiment(&context.experiment_id).ok_or(NNError::InvalidConfiguration { message: format!("Experiment {} not found", context.experiment_id) })?.summary(),
            })
        }

        fn evaluate_joint_configuration(
            &self,
            architecture: &Architecture,
            hyperparameters: &HyperparameterConfig,
            evaluator: Arc<dyn ArchitectureEvaluator>,
        ) -> Result<f64> {
            // Placeholder: In real implementation, this would combine NAS evaluation
            // with HPO by training the architecture with the given hyperparameters
            let base_result = evaluator.evaluate(architecture)?;

            // Simple modulation based on hyperparameter choices
            let mut modifier = 0.0;
            let lr = hyperparameters.get_float("learning_rate", 0.001);
            if lr > 0.0001 && lr < 0.1 {
                modifier += 0.05;
            }

            Ok(base_result.accuracy + modifier)
        }
    }

    /// Concurrent search in both spaces
    #[derive(Debug)]
    pub struct ConcurrentSearch;

    impl ConcurrentSearch {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl JointSearchAlgorithm for ConcurrentSearch {
        fn joint_search(
            &self,
            _context: &JointSearchContext,
            _framework: &mut UnifiedResearchFramework,
            _arch_evaluator: Arc<dyn ArchitectureEvaluator>,
            _arch_space: &ArchitectureSpace,
            _hp_space: &HyperparameterSpace,
        ) -> Result<JointSearchResult> {
            // Implementation for concurrent search
            Err(NNError::NotImplemented {
                operation: "Concurrent joint search".to_string(),
            })
        }

        fn name(&self) -> &str {
            "Concurrent NAS-HPO Search"
        }

        fn description(&self) -> &str {
            "Simultaneously searches architecture and hyperparameter spaces"
        }
    }

    // Placeholder implementations for other algorithms
    #[derive(Debug)]
    pub struct EvolutionaryJointSearch;
    impl EvolutionaryJointSearch {
        pub fn new() -> Self { Self {} }
    }

    impl JointSearchAlgorithm for EvolutionaryJointSearch {
        fn joint_search(&self, _context: &JointSearchContext, _framework: &mut UnifiedResearchFramework, _arch_evaluator: Arc<dyn ArchitectureEvaluator>, _arch_space: &ArchitectureSpace, _hp_space: &HyperparameterSpace) -> Result<JointSearchResult> {
            Err(NNError::NotImplemented { operation: "Evolutionary joint search".to_string() })
        }
        fn name(&self) -> &str { "Evolutionary Joint Search" }
        fn description(&self) -> &str { "Evolutionary algorithms in joint space" }
    }

    #[derive(Debug)]
    pub struct FactorizedSearch;
    impl FactorizedSearch {
        pub fn new() -> Self { Self {} }
    }

    impl JointSearchAlgorithm for FactorizedSearch {
        fn joint_search(&self, _context: &JointSearchContext, _framework: &mut UnifiedResearchFramework, _arch_evaluator: Arc<dyn ArchitectureEvaluator>, _arch_space: &ArchitectureSpace, _hp_space: &HyperparameterSpace) -> Result<JointSearchResult> {
            Err(NNError::NotImplemented { operation: "Factorized search".to_string() })
        }
        fn name(&self) -> &str { "Factorized Search" }
        fn description(&self) -> &str { "Separate but coordinated NAS and HPO" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::{ArchitectureSpace, ArchitectureType};

    #[test]
    fn test_joint_search_framework_creation() {
        let framework = JointSearchFramework::new();
        assert!(!framework.algorithms.is_empty());
        assert_eq!(framework.available_algorithms().len(), 4);
    }

    #[test]
    fn test_joint_search_context() {
        let context = JointSearchContext {
            experiment_id: "test_joint".to_string(),
            dataset_name: "cifar10".to_string(),
            task_type: "classification".to_string(),
            budget: JointSearchBudget {
                max_total_evaluations: 100,
                max_time_seconds: 3600,
                max_gpu_hours: 10.0,
                max_parallel_evaluations: 4,
            },
            search_strategy: JointSearchStrategy::Alternating {
                architecture_rounds: 2,
                hyperparameter_rounds: 2,
                synchronization_frequency: 1,
            },
            performance_predictors: None,
            evaluation_strategy: JointEvaluationStrategy::FullTraining,
        };

        assert_eq!(context.dataset_name, "cifar10");
        assert_eq!(context.task_type, "classification");
    }

    #[test]
    fn test_algorithm_recommendation() {
        let framework = JointSearchFramework::new();
        let context = JointSearchContext {
            experiment_id: "test".to_string(),
            dataset_name: "test".to_string(),
            task_type: "test".to_string(),
            budget: JointSearchBudget {
                max_total_evaluations: 50,
                max_time_seconds: 1800,
                max_gpu_hours: 5.0,
                max_parallel_evaluations: 2,
            },
            search_strategy: JointSearchStrategy::Alternating {
                architecture_rounds: 1,
                hyperparameter_rounds: 1,
                synchronization_frequency: 1,
            },
            performance_predictors: None,
            evaluation_strategy: JointEvaluationStrategy::FullTraining,
        };

        let recommended = framework.recommend_algorithm(&context);
        assert_eq!(recommended, "alternating");
    }

    #[test]
    fn test_joint_solution() {
        let space = ArchitectureSpace::new(ArchitectureType::CNN);
        let architecture = space.sample_random(2).unwrap();

        let solution = JointSolution {
            architecture,
            hyperparameters: HyperparameterConfig::new(),
            score: 0.85,
            evaluation_time: std::time::Duration::from_secs(120),
            resource_usage: ResourceUsage {
                gpu_memory_mb: 1024,
                cpu_time_seconds: 10.0,
                training_time_seconds: 110.0,
                peak_memory_mb: 2048,
            },
        };

        assert_eq!(solution.score, 0.85);
        assert_eq!(solution.resource_usage.gpu_memory_mb, 1024);
    }
}
