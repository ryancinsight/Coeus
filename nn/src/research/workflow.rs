//! Research Workflow Templates
//!
//! This module provides predefined workflow templates and custom workflow construction
//! for coordinating research experiments across multiple agents.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use super::ResearchDomain;

/// Research workflow definition
#[derive(Debug, Clone)]
pub struct ResearchWorkflow {
    /// Workflow identifier
    pub id: String,
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: String,
    /// Target research domain
    pub domain: ResearchDomain,
    /// Workflow steps
    pub steps: Vec<WorkflowStep>,
    /// Workflow parameters
    pub parameters: HashMap<String, Value>,
    /// Execution constraints
    pub constraints: WorkflowConstraints,
}

/// Single workflow step
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Step identifier
    pub id: String,
    /// Step name
    pub name: String,
    /// Agent type to execute this step
    pub agent_type: String,
    /// Step configuration
    pub config: Value,
    /// Dependencies (step IDs this step depends on)
    pub dependencies: Vec<String>,
    /// Step priority
    pub priority: u32,
}

/// Workflow execution constraints
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkflowConstraints {
    /// Maximum execution time (seconds)
    pub max_execution_time: Option<u64>,
    /// Maximum resource usage
    pub resource_limits: Option<HashMap<String, f64>>,
    /// Quality requirements
    pub quality_thresholds: HashMap<String, f64>,
}


/// Predefined workflow templates
pub struct WorkflowTemplate;

impl WorkflowTemplate {
    /// Create NAS-HPO collaborative workflow
    pub fn nas_hpo_collaboration(objective: &str) -> ResearchWorkflow {
        ResearchWorkflow {
            id: "nas_hpo_collaboration".to_string(),
            name: "NAS-HPO Collaborative Optimization".to_string(),
            description: "Joint neural architecture search and hyperparameter optimization".to_string(),
            domain: ResearchDomain::AutoML,
            steps: vec![
                WorkflowStep {
                    id: "nas_initial".to_string(),
                    name: "Initial NAS Exploration".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({
                        "algorithm": "darts",
                        "budget": 50,
                        "objective": objective
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "hpo_refinement".to_string(),
                    name: "HPO Refinement".to_string(),
                    agent_type: "hpo".to_string(),
                    config: serde_json::json!({
                        "algorithm": "bayesian",
                        "budget": 100,
                        "use_nas_insights": true
                    }),
                    dependencies: vec!["nas_initial".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "meta_learning".to_string(),
                    name: "Meta-Learning Adaptation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithm": "maml",
                        "adapt_steps": 5,
                        "use_previous_insights": true
                    }),
                    dependencies: vec!["hpo_refinement".to_string()],
                    priority: 8,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(3600),
                ..Default::default()
            },
        }
    }

    /// Create comprehensive AutoML pipeline
    pub fn comprehensive_automl() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "comprehensive_automl".to_string(),
            name: "Comprehensive AutoML".to_string(),
            description: "End-to-end automated machine learning pipeline".to_string(),
            domain: ResearchDomain::AutoML,
            steps: vec![
                WorkflowStep {
                    id: "data_analysis".to_string(),
                    name: "Dataset Analysis".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({"task": "analysis"}),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "architecture_search".to_string(),
                    name: "Architecture Search".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({"budget": 100}),
                    dependencies: vec!["data_analysis".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "hpo_optimization".to_string(),
                    name: "Hyperparameter Optimization".to_string(),
                    agent_type: "hpo".to_string(),
                    config: serde_json::json!({"algorithms": ["bayesian", "population"]}),
                    dependencies: vec!["architecture_search".to_string()],
                    priority: 8,
                },
                WorkflowStep {
                    id: "meta_fine_tuning".to_string(),
                    name: "Meta-Learning Fine-tuning".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({"episodes": 50}),
                    dependencies: vec!["hpo_optimization".to_string()],
                    priority: 7,
                },
                WorkflowStep {
                    id: "validation".to_string(),
                    name: "Cross-validation".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({"folds": 5, "task": "validation"}),
                    dependencies: vec!["meta_fine_tuning".to_string()],
                    priority: 6,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(7200),
                quality_thresholds: {
                    let mut thresholds = HashMap::new();
                    thresholds.insert("accuracy".to_string(), 0.9);
                    thresholds
                },
                ..Default::default()
            },
        }
    }

    /// Create comparative benchmark workflow
    pub fn comparative_benchmark() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "comparative_benchmark".to_string(),
            name: "Comparative Algorithm Benchmark".to_string(),
            description: "Compare different optimization algorithms on standard benchmarks".to_string(),
            domain: ResearchDomain::GeneralML,
            steps: vec![
                WorkflowStep {
                    id: "nas_benchmark".to_string(),
                    name: "NAS Algorithm Comparison".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({
                        "algorithms": ["darts", "reinforcement", "evolutionary"],
                        "benchmarks": ["cifar10", "imagenet_tiny"]
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "hpo_benchmark".to_string(),
                    name: "HPO Algorithm Comparison".to_string(),
                    agent_type: "hpo".to_string(),
                    config: serde_json::json!({
                        "algorithms": ["bayesian", "population", "bandits"],
                        "functions": ["rosenbrock", "sphere", "rastrigin"]
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "meta_benchmark".to_string(),
                    name: "Meta-Learning Comparison".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithms": ["maml", "prototypical", "relationnet"],
                        "datasets": ["omniglot", "miniimagenet"]
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "results_analysis".to_string(),
                    name: "Comparative Analysis".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({"task": "analysis"}),
                    dependencies: vec![
                        "nas_benchmark".to_string(),
                        "hpo_benchmark".to_string(),
                        "meta_benchmark".to_string()
                    ],
                    priority: 5,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(14400), // 4 hours
                ..Default::default()
            },
        }
    }

    /// Create few-shot learning pipeline workflow
    pub fn few_shot_learning_pipeline() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "few_shot_learning_pipeline".to_string(),
            name: "Few-Shot Learning Research Pipeline".to_string(),
            description: "Complete pipeline for few-shot learning research with meta-training and adaptation".to_string(),
            domain: ResearchDomain::MetaLearning,
            steps: vec![
                WorkflowStep {
                    id: "dataset_preparation".to_string(),
                    name: "Few-Shot Dataset Preparation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "dataset_prep",
                        "n_way": 5,
                        "k_shot": 1,
                        "n_query": 15,
                        "num_classes": 20
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "maml_meta_training".to_string(),
                    name: "MAML Meta-Training".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "experiment_type": "meta_training",
                        "algorithm": "maml",
                        "tasks_per_step": 4,
                        "num_iterations": 100,
                        "inner_lr": 0.01,
                        "outer_lr": 0.001
                    }),
                    dependencies: vec!["dataset_preparation".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "prototypical_training".to_string(),
                    name: "Prototypical Networks Training".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "experiment_type": "few_shot_learning",
                        "algorithm": "prototypical",
                        "num_episodes": 50,
                        "n_way": 5,
                        "k_shot": 1,
                        "adaptation_steps": 5
                    }),
                    dependencies: vec!["dataset_preparation".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "few_shot_evaluation".to_string(),
                    name: "Few-Shot Adaptation Evaluation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "evaluation",
                        "adaptation_shots": [1, 5, 10],
                        "test_episodes": 100
                    }),
                    dependencies: vec![
                        "maml_meta_training".to_string(),
                        "prototypical_training".to_string()
                    ],
                    priority: 8,
                },
                WorkflowStep {
                    id: "domain_adaptation".to_string(),
                    name: "Cross-Domain Adaptation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "domain_transfer",
                        "source_domains": ["synthetic"],
                        "target_domains": ["real_world"],
                        "adaptation_budget": 50
                    }),
                    dependencies: vec!["few_shot_evaluation".to_string()],
                    priority: 7,
                },
                WorkflowStep {
                    id: "results_synthesis".to_string(),
                    name: "Results Synthesis and Insights".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "synthesis",
                        "metrics": ["accuracy", "adaptation_efficiency", "generalization"],
                        "visualizations": true
                    }),
                    dependencies: vec!["domain_adaptation".to_string()],
                    priority: 6,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(7200), // 2 hours
                quality_thresholds: {
                    let mut thresholds = HashMap::new();
                    thresholds.insert("few_shot_accuracy".to_string(), 0.8);
                    thresholds.insert("adaptation_efficiency".to_string(), 0.7);
                    thresholds
                },
                ..Default::default()
            },
        }
    }

    /// Create continual learning research workflow
    pub fn continual_learning_research() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "continual_learning_research".to_string(),
            name: "Continual Learning Research Pipeline".to_string(),
            description: "Research pipeline for continual learning with meta-learning approaches".to_string(),
            domain: ResearchDomain::MetaLearning,
            steps: vec![
                WorkflowStep {
                    id: "task_sequence_generation".to_string(),
                    name: "Task Sequence Generation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "sequence_gen",
                        "num_tasks": 10,
                        "task_complexity": "increasing",
                        "domain_shifts": ["small", "medium", "large"]
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "baseline_training".to_string(),
                    name: "Baseline Continual Learning".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithm": "baseline",
                        "replay_buffer": true,
                        "regularization": "l2",
                        "buffer_size": 1000
                    }),
                    dependencies: vec!["task_sequence_generation".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "maml_continual".to_string(),
                    name: "MAML for Continual Learning".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithm": "maml_continual",
                        "meta_update_frequency": 10,
                        "task_memory": 5,
                        "plasticity_stability_tradeoff": 0.1
                    }),
                    dependencies: vec!["task_sequence_generation".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "memory_management".to_string(),
                    name: "Intelligent Memory Management".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "memory_optimization",
                        "selection_strategy": "importance_weighted",
                        "compression": "quantization",
                        "retrieval_mechanism": "attention_based"
                    }),
                    dependencies: vec!["baseline_training".to_string()],
                    priority: 8,
                },
                WorkflowStep {
                    id: "catastrophic_forgetting_analysis".to_string(),
                    name: "Catastrophic Forgetting Analysis".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "forgetting_analysis",
                        "metrics": ["backward_transfer", "forward_transfer", "plasticity_loss"],
                        "granularity": "task_level"
                    }),
                    dependencies: vec![
                        "baseline_training".to_string(),
                        "maml_continual".to_string()
                    ],
                    priority: 7,
                },
                WorkflowStep {
                    id: "knowledge_transfer_study".to_string(),
                    name: "Knowledge Transfer Study".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "transfer_analysis",
                        "transfer_metrics": ["positive_transfer", "negative_transfer", "neutral_transfer"],
                        "cross_task_similarities": true
                    }),
                    dependencies: vec!["catastrophic_forgetting_analysis".to_string()],
                    priority: 6,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(14400), // 4 hours
                quality_thresholds: {
                    let mut thresholds = HashMap::new();
                    thresholds.insert("catastrophic_forgetting".to_string(), 0.3); // Max 30% forgetting
                    thresholds.insert("knowledge_retention".to_string(), 0.7); // Min 70% retention
                    thresholds
                },
                ..Default::default()
            },
        }
    }

    /// Create meta-learning benchmark orchestration workflow
    pub fn meta_learning_benchmark_orchestration() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "meta_learning_benchmark_orchestration".to_string(),
            name: "Meta-Learning Benchmark Suite".to_string(),
            description: "Comprehensive benchmark suite for meta-learning algorithms and setups".to_string(),
            domain: ResearchDomain::MetaLearning,
            steps: vec![
                WorkflowStep {
                    id: "benchmark_dataset_setup".to_string(),
                    name: "Benchmark Dataset Configuration".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "datasets": ["omniglot", "miniimagenet", "tieredimagenet", "cifar_fs", "fc100"],
                        "splits": ["train", "val", "test"],
                        "standardization": true
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "maml_benchmarking".to_string(),
                    name: "MAML Algorithm Benchmarking".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithm": "maml",
                        "variants": ["first_order", "second_order", "reptile"],
                        "hyperparameter_sweep": {
                            "inner_lr": [0.01, 0.1, 1.0],
                            "outer_lr": [0.001, 0.01],
                            "num_inner_steps": [1, 5, 10]
                        }
                    }),
                    dependencies: vec!["benchmark_dataset_setup".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "prototypical_benchmarking".to_string(),
                    name: "Prototypical Networks Benchmarking".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "algorithm": "prototypical",
                        "encoders": ["conv4", "conv6", "resnet"],
                        "distance_metrics": ["euclidean", "cosine", "learned"],
                        "hyperparameter_sweep": {
                            "n_way": [5, 10, 20],
                            "k_shot": [1, 5]
                        }
                    }),
                    dependencies: vec!["benchmark_dataset_setup".to_string()],
                    priority: 9,
                },
                WorkflowStep {
                    id: "few_shot_settings_benchmark".to_string(),
                    name: "Few-Shot Settings Benchmark".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "settings": [
                            {"n_way": 5, "k_shot": 1, "n_query": 15},
                            {"n_way": 5, "k_shot": 5, "n_query": 10},
                            {"n_way": 10, "k_shot": 1, "n_query": 10}
                        ],
                        "num_episodes_per_setting": 1000
                    }),
                    dependencies: vec![
                        "maml_benchmarking".to_string(),
                        "prototypical_benchmarking".to_string()
                    ],
                    priority: 8,
                },
                WorkflowStep {
                    id: "statistical_analysis".to_string(),
                    name: "Statistical Analysis and Significance Testing".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "tests": ["paired_t_test", "wilcoxon", "bootstrap"],
                        "confidence_level": 0.95,
                        "multiple_comparison_correction": "bonferroni"
                    }),
                    dependencies: vec!["few_shot_settings_benchmark".to_string()],
                    priority: 7,
                },
                WorkflowStep {
                    id: "benchmark_report_generation".to_string(),
                    name: "Benchmark Report Generation".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "report_format": "comprehensive",
                        "include_raw_data": true,
                        "generate_plots": true,
                        "comparison_tables": true
                    }),
                    dependencies: vec!["statistical_analysis".to_string()],
                    priority: 6,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(21600), // 6 hours
                ..Default::default()
            },
        }
    }

    /// Create cross-agent knowledge transfer workflow (Meta-Learning with NAS/HPO)
    pub fn cross_agent_meta_learning_workflow() -> ResearchWorkflow {
        ResearchWorkflow {
            id: "cross_agent_meta_learning".to_string(),
            name: "Cross-Agent Meta-Learning Pipeline".to_string(),
            description: "Meta-learning enhanced by knowledge transfer from NAS and HPO agents".to_string(),
            domain: ResearchDomain::AutoML,
            steps: vec![
                WorkflowStep {
                    id: "nas_architecture_search".to_string(),
                    name: "NAS Architecture Exploration".to_string(),
                    agent_type: "nas".to_string(),
                    config: serde_json::json!({
                        "budget": 100,
                        "search_space": "meta_learning_optimized",
                        "diversity_objective": true
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "hpo_hyperparameter_search".to_string(),
                    name: "HPO for Meta-Learning Hyperparameters".to_string(),
                    agent_type: "hpo".to_string(),
                    config: serde_json::json!({
                        "parameters": ["inner_lr", "outer_lr", "batch_size", "num_inner_steps"],
                        "budget": 200,
                        "use_meta_objective": true
                    }),
                    dependencies: vec![],
                    priority: 10,
                },
                WorkflowStep {
                    id: "meta_learning_with_insights".to_string(),
                    name: "Meta-Learning with Cross-Agent Insights".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "use_nas_insights": true,
                        "use_hpo_insights": true,
                        "adaptation_strategy": "insight_driven",
                        "num_meta_iterations": 200
                    }),
                    dependencies: vec![
                        "nas_architecture_search".to_string(),
                        "hpo_hyperparameter_search".to_string()
                    ],
                    priority: 9,
                },
                WorkflowStep {
                    id: "transfer_effectiveness_analysis".to_string(),
                    name: "Knowledge Transfer Effectiveness Analysis".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "task": "transfer_analysis",
                        "compare_with_baseline": true,
                        "attribution_analysis": true
                    }),
                    dependencies: vec!["meta_learning_with_insights".to_string()],
                    priority: 8,
                },
                WorkflowStep {
                    id: "collaborative_optimization".to_string(),
                    name: "Collaborative Multi-Agent Optimization".to_string(),
                    agent_type: "meta".to_string(),
                    config: serde_json::json!({
                        "coordination_strategy": "iterative_refinement",
                        "max_iterations": 10,
                        "convergence_threshold": 0.01
                    }),
                    dependencies: vec!["transfer_effectiveness_analysis".to_string()],
                    priority: 7,
                },
            ],
            parameters: HashMap::new(),
            constraints: WorkflowConstraints {
                max_execution_time: Some(10800), // 3 hours
                quality_thresholds: {
                    let mut thresholds = HashMap::new();
                    thresholds.insert("transfer_improvement".to_string(), 0.05); // 5% improvement minimum
                    thresholds.insert("collaboration_efficiency".to_string(), 0.8);
                    thresholds
                },
                ..Default::default()
            },
        }
    }
}

/// Configuration-driven workflow specification
/// Supports declarative workflow definition via YAML/JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSpec {
    /// Workflow metadata
    pub metadata: WorkflowMetadata,
    /// Workflow steps specification
    pub steps: Vec<StepSpec>,
    /// Workflow constraints and configuration
    pub config: WorkflowConfig,
    /// Optional template inheritance
    #[serde(default)]
    pub extends: Option<String>,
}

/// Workflow metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMetadata {
    /// Workflow unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Workflow description
    pub description: String,
    /// Research domain
    pub domain: ResearchDomain,
    /// Workflow version
    #[serde(default)]
    pub version: String,
    /// Workflow author/contributor
    #[serde(default)]
    pub author: String,
    /// Workflow tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Step specification for declarative workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepSpec {
    /// Step unique identifier within workflow
    pub id: String,
    /// Step name
    pub name: String,
    /// Agent type to execute this step
    pub agent_type: String,
    /// Step configuration
    #[serde(default)]
    pub config: HashMap<String, Value>,
    /// Dependencies (step IDs this step depends on)
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Step priority (higher = more important)
    #[serde(default = "default_priority")]
    pub priority: u32,
    /// Resource requirements
    #[serde(default)]
    pub resources: ResourceRequirements,
    /// Retry configuration
    #[serde(default)]
    pub retry: RetryConfig,
    /// Conditional execution
    #[serde(default)]
    pub condition: Option<String>,
}

/// Resource requirements for workflow steps
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRequirements {
    /// Required GPU count
    #[serde(default)]
    pub gpu_required: usize,
    /// Required CPU cores
    #[serde(default = "default_cpu_required")]
    pub cpu_required: usize,
    /// Required memory in MB
    #[serde(default = "default_memory_mb")]
    pub memory_mb: usize,
    /// Maximum execution time in seconds
    #[serde(default)]
    pub max_execution_time: Option<u64>,
}

/// Retry configuration for failed steps
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetryConfig {
    /// Maximum retry attempts
    #[serde(default)]
    pub max_attempts: usize,
    /// Delay between retries (seconds)
    #[serde(default = "default_retry_delay")]
    pub delay_seconds: u64,
    /// Exponential backoff multiplier
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Execution constraints
    #[serde(default)]
    pub constraints: WorkflowConstraints,
    /// Global parameters available to all steps
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
    /// Execution mode
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    /// Failure handling strategy
    #[serde(default)]
    pub failure_strategy: FailureStrategy,
}

/// Execution modes for workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Sequential execution
    Sequential,
    /// Parallel execution where possible
    Parallel,
    /// Adaptive execution based on results
    Adaptive,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Parallel
    }
}

/// Failure handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureStrategy {
    /// Stop on first failure
    FailFast,
    /// Continue with remaining steps
    ContinueOnFailure,
    /// Retry failed steps
    RetryFailed,
}

impl Default for FailureStrategy {
    fn default() -> Self {
        FailureStrategy::FailFast
    }
}

/// Workflow loader for declarative specifications
pub struct WorkflowLoader;

impl WorkflowLoader {
    /// Load workflow from YAML file
    pub fn load_from_yaml<P: AsRef<Path>>(path: P) -> crate::error::Result<ResearchWorkflow> {
        let content = fs::read_to_string(path)?;
        let spec: WorkflowSpec = serde_yaml::from_str(&content)
            .map_err(|e| crate::error::NNError::InvalidConfiguration {
                message: format!("Failed to parse workflow YAML: {}", e),
            })?;

        Self::spec_to_workflow(spec)
    }

    /// Load workflow from JSON file
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> crate::error::Result<ResearchWorkflow> {
        let content = fs::read_to_string(path)?;
        let spec: WorkflowSpec = serde_json::from_str(&content)
            .map_err(|e| crate::error::NNError::InvalidConfiguration {
                message: format!("Failed to parse workflow JSON: {}", e),
            })?;

        Self::spec_to_workflow(spec)
    }

    /// Convert workflow specification to ResearchWorkflow
    pub fn spec_to_workflow(spec: WorkflowSpec) -> crate::error::Result<ResearchWorkflow> {
        let steps = spec.steps.into_iter()
            .map(|step_spec| {
                // Convert step config to JSON Value
                let config = serde_json::to_value(step_spec.config)
                    .unwrap_or(Value::Object(serde_json::Map::new()));

                WorkflowStep {
                    id: step_spec.id,
                    name: step_spec.name,
                    agent_type: step_spec.agent_type,
                    config,
                    dependencies: step_spec.depends_on,
                    priority: step_spec.priority,
                }
            })
            .collect();

        // Merge global parameters with constraints
        let mut constraints = spec.config.constraints;
        constraints.resource_limits = Some({
            let mut limits = HashMap::new();
            // Add global resource limits if specified
            limits
        });

        Ok(ResearchWorkflow {
            id: spec.metadata.id,
            name: spec.metadata.name,
            description: spec.metadata.description,
            domain: spec.metadata.domain,
            steps,
            parameters: spec.config.parameters,
            constraints,
        })
    }

    /// Save workflow specification to YAML file
    pub fn save_to_yaml<P: AsRef<Path>>(workflow: &ResearchWorkflow, path: P) -> crate::error::Result<()> {
        let spec = Self::workflow_to_spec(workflow)?;
        let yaml = serde_yaml::to_string(&spec)
            .map_err(|e| crate::error::NNError::ExecutionError {
                message: format!("Failed to serialize workflow to YAML: {}", e),
            })?;
        fs::write(path, yaml)?;
        Ok(())
    }

    /// Convert ResearchWorkflow to specification
    pub fn workflow_to_spec(workflow: &ResearchWorkflow) -> crate::error::Result<WorkflowSpec> {
        let steps = workflow.steps.iter()
            .map(|step| {
                let config = serde_json::from_value(step.config.clone())
                    .unwrap_or_default();

                StepSpec {
                    id: step.id.clone(),
                    name: step.name.clone(),
                    agent_type: step.agent_type.clone(),
                    config,
                    depends_on: step.dependencies.clone(),
                    priority: step.priority,
                    resources: ResourceRequirements::default(),
                    retry: RetryConfig::default(),
                    condition: None,
                }
            })
            .collect();

        let metadata = WorkflowMetadata {
            id: workflow.id.clone(),
            name: workflow.name.clone(),
            description: workflow.description.clone(),
            domain: workflow.domain.clone(),
            version: "1.0.0".to_string(),
            author: "auto-generated".to_string(),
            tags: Vec::new(),
        };

        let config = WorkflowConfig {
            constraints: workflow.constraints.clone(),
            parameters: workflow.parameters.clone(),
            execution_mode: ExecutionMode::Parallel,
            failure_strategy: FailureStrategy::FailFast,
        };

        Ok(WorkflowSpec {
            metadata,
            steps,
            config,
            extends: None,
        })
    }
}

// Default value functions
fn default_priority() -> u32 { 5 }
fn default_cpu_required() -> usize { 1 }
fn default_memory_mb() -> usize { 1024 }
fn default_retry_delay() -> u64 { 60 }
fn default_backoff_multiplier() -> f64 { 2.0 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_spec_serialization() {
        let spec = WorkflowSpec {
            metadata: WorkflowMetadata {
                id: "test_workflow".to_string(),
                name: "Test Workflow".to_string(),
                description: "A test workflow".to_string(),
                domain: ResearchDomain::AutoML,
                version: "1.0.0".to_string(),
                author: "test".to_string(),
                tags: vec!["test".to_string()],
            },
            steps: vec![
                StepSpec {
                    id: "step1".to_string(),
                    name: "Step 1".to_string(),
                    agent_type: "nas".to_string(),
                    config: HashMap::new(),
                    depends_on: vec![],
                    priority: 10,
                    resources: ResourceRequirements::default(),
                    retry: RetryConfig::default(),
                    condition: None,
                }
            ],
            config: WorkflowConfig {
                constraints: WorkflowConstraints::default(),
                parameters: HashMap::new(),
                execution_mode: ExecutionMode::Parallel,
                failure_strategy: FailureStrategy::FailFast,
            },
            extends: None,
        };

        // Test JSON serialization
        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: WorkflowSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.metadata.id, spec.metadata.id);
    }

    #[test]
    fn test_workflow_conversion() {
        let workflow = WorkflowTemplate::nas_hpo_collaboration("accuracy");
        let spec = WorkflowLoader::workflow_to_spec(&workflow).unwrap();
        let converted_back = WorkflowLoader::spec_to_workflow(spec).unwrap();

        assert_eq!(converted_back.id, workflow.id);
        assert_eq!(converted_back.steps.len(), workflow.steps.len());
    }
}
