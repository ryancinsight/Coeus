//! Research Workflow Templates
//!
//! This module provides predefined workflow templates and custom workflow construction
//! for coordinating research experiments across multiple agents.

use serde_json::Value;
use std::collections::HashMap;
use crate::error::Result;
use super::{ExperimentSpec, ResearchDomain};

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
#[derive(Debug, Clone)]
pub struct WorkflowConstraints {
    /// Maximum execution time (seconds)
    pub max_execution_time: Option<u64>,
    /// Maximum resource usage
    pub resource_limits: Option<HashMap<String, f64>>,
    /// Quality requirements
    pub quality_thresholds: HashMap<String, f64>,
}

impl Default for WorkflowConstraints {
    fn default() -> Self {
        Self {
            max_execution_time: None,
            resource_limits: None,
            quality_thresholds: HashMap::new(),
        }
    }
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
}
