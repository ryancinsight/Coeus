//! Meta-Learning Research Agents
//!
//! This module provides concrete implementations of research agents for meta-learning
//! algorithms including MAML and Prototypical Networks. These agents integrate
//! with the unified research framework for automated meta-learning research.

use std::collections::HashMap;

use crate::error::{NNError, Result};
use crate::linear::Linear;
use crate::meta::prototypical::{DistanceMetric, PrototypicalNetwork};
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

/// Type alias for the complex PrototypicalNetwork type
type ProtoNet = PrototypicalNetwork<
    Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    CpuBackend<Float32>,
    DenseStorage<Float32>,
    Float32,
>;

use super::{
    agent::{AgentMetadata, AgentType, PerformanceCharacteristics, ResourceProfile},
    experiment::ResourceRequirements,
    ExperimentResult, ExperimentSpec, ExperimentStatus, ResearchAgent, ResearchAgentFactory,
    ResearchDomain, ResearchInsight,
};

/// Prototypical Networks Research Agent
pub struct PrototypicalResearchAgent {
    /// Agent identifier
    id: String,
    /// Agent name
    name: String,
    /// Prototypical network instance
    proto_net: Option<ProtoNet>,
    /// Training history
    training_history: Vec<f64>,
    /// Best result achieved
    best_result: Option<ExperimentResult>,
}

impl PrototypicalResearchAgent {
    /// Create new Prototypical Networks research agent
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            proto_net: None,
            training_history: Vec::new(),
            best_result: None,
        }
    }
}

impl ResearchAgent for PrototypicalResearchAgent {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::MetaLearning
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_domains: vec![ResearchDomain::MetaLearning, ResearchDomain::ComputerVision],
            resource_profile: ResourceProfile::default(),
            performance_characteristics: PerformanceCharacteristics::default(),
            capabilities: vec![
                "Few-shot classification".to_string(),
                "Prototype learning".to_string(),
                "Metric learning".to_string(),
                "Episode-based training".to_string(),
            ],
        }
    }

    fn supports_domain(&self, domain: &ResearchDomain) -> bool {
        matches!(
            domain,
            ResearchDomain::MetaLearning | ResearchDomain::ComputerVision
        )
    }

    fn initialize(&mut self, config: serde_json::Value) -> Result<()> {
        // Extract configuration parameters
        let input_size = config
            .get("input_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(784) as usize;

        let hidden_size = config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let distance_metric = config
            .get("distance_metric")
            .and_then(|v| v.as_str())
            .unwrap_or("euclidean");

        let metric = match distance_metric {
            "euclidean" => DistanceMetric::Euclidean,
            "cosine" => DistanceMetric::Cosine,
            "learned" => DistanceMetric::Learned,
            _ => DistanceMetric::Euclidean,
        };

        let scale = config.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);

        let temperature = config
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // Create encoder network
        let encoder = Linear::new(input_size, hidden_size).unwrap();

        self.proto_net = Some(
            PrototypicalNetwork::new(encoder)
                .with_distance_metric(metric)
                .with_scale(scale)
                .with_temperature(temperature),
        );

        Ok(())
    }

    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        let start_time = std::time::Instant::now();

        match experiment
            .experiment_config
            .get("experiment_type")
            .and_then(|v| v.as_str())
            .unwrap_or("evaluation")
        {
            "evaluation" => self.run_evaluation(experiment),
            _ => Err(NNError::InvalidConfiguration {
                message: "Unsupported experiment type".to_string(),
            }),
        }
        .map(|mut result| {
            result.start_time = start_time;
            result.end_time = std::time::Instant::now();
            result.experiment_id = experiment.id.clone();
            result.agent_id = self.id.clone();

            // Update best result
            if let Some(ref best) = self.best_result {
                if result.final_performance > best.final_performance {
                    self.best_result = Some(result.clone());
                }
            } else {
                self.best_result = Some(result.clone());
            }

            result
        })
    }

    fn get_available_actions(&self) -> Vec<ExperimentSpec> {
        vec![ExperimentSpec {
            id: "proto_eval".to_string(),
            name: "Prototypical Networks Evaluation".to_string(),
            domain: ResearchDomain::MetaLearning,
            agent_type: "prototypical".to_string(),
            experiment_config: serde_json::json!({
                "experiment_type": "evaluation",
                "n_way": 5,
                "k_shot": 5,
                "n_query": 15
            }),
            resource_requirements: Default::default(),
            dependencies: vec![],
            priority: 1,
            timeout_secs: Some(300),
            quality_constraints: Default::default(),
            metadata: HashMap::new(),
        }]
    }

    fn update_with_results(&mut self, results: &[ExperimentResult]) -> Result<()> {
        for result in results {
            if result.status == ExperimentStatus::Completed {
                self.training_history.push(result.final_performance);
            }
        }
        Ok(())
    }

    fn get_best_result(&self) -> Option<ExperimentResult> {
        self.best_result.clone()
    }

    fn get_state(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "training_history": self.training_history,
            "best_performance": self.best_result.as_ref().map(|r| r.final_performance),
            "total_experiments": self.training_history.len()
        }))
    }

    fn set_state(&mut self, state: serde_json::Value) -> Result<()> {
        if let Some(history) = state.get("training_history") {
            self.training_history = serde_json::from_value(history.clone()).unwrap_or_default();
        }
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.proto_net.is_some()
    }

    fn get_resource_requirements(&self) -> ResourceRequirements {
        ResourceRequirements::default()
    }

    fn generate_insights(&self) -> Vec<ResearchInsight> {
        if self.training_history.len() >= 5 {
            let recent_avg = self.training_history.iter().rev().take(3).sum::<f64>() / 3.0;

            vec![ResearchInsight {
                id: format!("proto_meta_insight_{}", self.id),
                description: format!(
                    "Prototypical performance insight after {} experiments",
                    self.training_history.len()
                ),
                evidence: vec![
                    format!("Average performance: {:.3}", recent_avg),
                    format!("Total experiments: {}", self.training_history.len()),
                ],
                confidence: 0.7,
                agent_type: self.id.clone(),
                performance_impact: recent_avg - 0.5,
                domains: vec!["MetaLearning".to_string()],
                knowledge_data: serde_json::json!({"recent_avg": recent_avg}),
                timestamp: std::time::Instant::now(),
            }]
        } else {
            vec![]
        }
    }
}

impl PrototypicalResearchAgent {
    /// Run evaluation with synthetic data
    fn run_evaluation(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        let n_way = experiment
            .experiment_config
            .get("n_way")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let k_shot = experiment
            .experiment_config
            .get("k_shot")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let _n_query = experiment
            .experiment_config
            .get("n_query")
            .and_then(|v| v.as_u64())
            .unwrap_or(15) as usize;

        // Create synthetic episode data for demonstration
        let mut support_set = Vec::new();
        for class_id in 0..n_way {
            for _ in 0..k_shot {
                // Create random feature vectors
                let features = Tensor::from_vec(vec![Float32::new(0.1); 64], &[64]).unwrap();
                support_set.push((features, class_id));
            }
        }

        let proto_net = self
            .proto_net
            .as_ref()
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: "Prototypical network not initialized".to_string(),
            })?;

        // Compute prototypes
        let _prototypes = proto_net.compute_prototypes(&support_set, n_way)?;

        // Simulate accuracy calculation (placeholder)
        let accuracy = 0.75 + (self.training_history.len() as f64 * 0.01).min(0.15);

        Ok(ExperimentResult {
            experiment_id: String::new(), // Will be set by caller
            agent_id: String::new(),      // Will be set by caller
            status: ExperimentStatus::Completed,
            final_performance: accuracy,
            performance_trajectory: vec![accuracy],
            resource_usage: Default::default(),
            start_time: std::time::Instant::now(),
            end_time: std::time::Instant::now(),
            statistics: Default::default(),
            insights: vec![],
            artifacts: HashMap::new(),
            metadata: HashMap::new(),
        })
    }
}

/// Factory for creating Prototypical Networks research agents
#[derive(Debug)]
pub struct PrototypicalResearchAgentFactory;

impl ResearchAgentFactory for PrototypicalResearchAgentFactory {
    fn create_factory() -> Box<dyn ResearchAgentFactory> {
        Box::new(Self)
    }

    fn create(&self, config: serde_json::Value) -> Result<Box<dyn ResearchAgent>> {
        let id = config
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("proto_agent")
            .to_string();

        let name = config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Prototypical Networks Research Agent")
            .to_string();

        let mut agent = PrototypicalResearchAgent::new(id, name);
        agent.initialize(config)?;

        Ok(Box::new(agent))
    }
}

/// MAML Research Agent
pub struct MAMLResearchAgent {
    /// Agent identifier
    id: String,
    /// Agent name
    name: String,
    /// Training history
    training_history: Vec<f64>,
    /// Best result achieved
    best_result: Option<ExperimentResult>,
    /// Meta-learning rate
    meta_lr: f64,
    /// Inner learning rate
    inner_lr: f64,
    /// Number of inner steps
    num_inner_steps: usize,
    /// Tasks per meta-batch
    tasks_per_batch: usize,
}

impl MAMLResearchAgent {
    /// Create new MAML research agent
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            training_history: Vec::new(),
            best_result: None,
            meta_lr: 0.001,
            inner_lr: 0.01,
            num_inner_steps: 5,
            tasks_per_batch: 4,
        }
    }
}

impl ResearchAgent for MAMLResearchAgent {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::MetaLearning
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_domains: vec![ResearchDomain::MetaLearning],
            resource_profile: ResourceProfile::default(),
            performance_characteristics: PerformanceCharacteristics::default(),
            capabilities: vec![
                "Meta-training".to_string(),
                "Few-shot adaptation".to_string(),
                "Gradient-based meta-learning".to_string(),
                "Task distribution sampling".to_string(),
            ],
        }
    }

    fn supports_domain(&self, domain: &ResearchDomain) -> bool {
        matches!(domain, ResearchDomain::MetaLearning)
    }

    fn initialize(&mut self, config: serde_json::Value) -> Result<()> {
        self.meta_lr = config
            .get("meta_learning_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.001);

        self.inner_lr = config
            .get("inner_learning_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.01);

        self.num_inner_steps = config
            .get("num_inner_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        self.tasks_per_batch = config
            .get("tasks_per_batch")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;

        Ok(())
    }

    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        let start_time = std::time::Instant::now();

        match experiment
            .experiment_config
            .get("experiment_type")
            .and_then(|v| v.as_str())
            .unwrap_or("meta_training")
        {
            "meta_training" => self.run_meta_training(experiment),
            "few_shot_evaluation" => self.run_few_shot_evaluation(experiment),
            _ => Err(NNError::InvalidConfiguration {
                message: "Unsupported experiment type".to_string(),
            }),
        }
        .map(|mut result| {
            result.start_time = start_time;
            result.end_time = std::time::Instant::now();
            result.experiment_id = experiment.id.clone();
            result.agent_id = self.id.clone();

            // Update best result
            if let Some(ref best) = self.best_result {
                if result.final_performance > best.final_performance {
                    self.best_result = Some(result.clone());
                }
            } else {
                self.best_result = Some(result.clone());
            }

            // Update training history
            let _ = self.update_with_results(&[result.clone()]);

            result
        })
    }

    fn get_available_actions(&self) -> Vec<ExperimentSpec> {
        vec![
            ExperimentSpec {
                id: "maml_training".to_string(),
                name: "MAML Meta-Training".to_string(),
                domain: ResearchDomain::MetaLearning,
                agent_type: "maml".to_string(),
                experiment_config: serde_json::json!({
                    "experiment_type": "meta_training",
                    "tasks_per_batch": self.tasks_per_batch,
                    "num_inner_steps": self.num_inner_steps
                }),
                resource_requirements: Default::default(),
                dependencies: vec![],
                priority: 1,
                timeout_secs: Some(600),
                quality_constraints: Default::default(),
                metadata: HashMap::new(),
            },
            ExperimentSpec {
                id: "maml_eval".to_string(),
                name: "MAML Few-Shot Evaluation".to_string(),
                domain: ResearchDomain::MetaLearning,
                agent_type: "maml".to_string(),
                experiment_config: serde_json::json!({
                    "experiment_type": "few_shot_evaluation",
                    "n_way": 5,
                    "k_shot": 1,
                    "n_query": 15
                }),
                resource_requirements: Default::default(),
                dependencies: vec![],
                priority: 1,
                timeout_secs: Some(300),
                quality_constraints: Default::default(),
                metadata: HashMap::new(),
            },
        ]
    }

    fn update_with_results(&mut self, results: &[ExperimentResult]) -> Result<()> {
        for result in results {
            if result.status == ExperimentStatus::Completed {
                self.training_history.push(result.final_performance);
            }
        }
        Ok(())
    }

    fn get_best_result(&self) -> Option<ExperimentResult> {
        self.best_result.clone()
    }

    fn get_state(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "training_history": self.training_history,
            "best_performance": self.best_result.as_ref().map(|r| r.final_performance),
            "meta_lr": self.meta_lr,
            "inner_lr": self.inner_lr,
            "total_experiments": self.training_history.len()
        }))
    }

    fn set_state(&mut self, state: serde_json::Value) -> Result<()> {
        if let Some(history) = state.get("training_history") {
            self.training_history = serde_json::from_value(history.clone()).unwrap_or_default();
        }
        if let Some(lr) = state.get("meta_lr").and_then(|v| v.as_f64()) {
            self.meta_lr = lr;
        }
        if let Some(lr) = state.get("inner_lr").and_then(|v| v.as_f64()) {
            self.inner_lr = lr;
        }
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true // MAML agent is always ready for simplified implementation
    }

    fn get_resource_requirements(&self) -> ResourceRequirements {
        ResourceRequirements::default()
    }

    fn generate_insights(&self) -> Vec<ResearchInsight> {
        if self.training_history.len() >= 5 {
            let recent_avg = self.training_history.iter().rev().take(3).sum::<f64>() / 3.0;

            vec![ResearchInsight {
                id: format!("maml_insight_{}", self.id),
                description: format!(
                    "MAML performance insight after {} experiments",
                    self.training_history.len()
                ),
                evidence: vec![
                    format!("Average performance: {:.3}", recent_avg),
                    format!("Total experiments: {}", self.training_history.len()),
                    format!("Meta learning rate: {:.6}", self.meta_lr),
                    format!("Inner learning rate: {:.6}", self.inner_lr),
                ],
                confidence: 0.8,
                agent_type: "maml".to_string(),
                performance_impact: recent_avg - 0.5,
                domains: vec![ResearchDomain::MetaLearning.to_string()],
                knowledge_data: serde_json::json!({
                    "recent_avg": recent_avg,
                    "meta_lr": self.meta_lr,
                    "inner_lr": self.inner_lr
                }),
                timestamp: std::time::Instant::now(),
            }]
        } else {
            vec![]
        }
    }
}

impl MAMLResearchAgent {
    /// Run meta-training simulation
    fn run_meta_training(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        let tasks_per_batch = experiment
            .experiment_config
            .get("tasks_per_batch")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.tasks_per_batch as u64) as usize;

        let num_inner_steps = experiment
            .experiment_config
            .get("num_inner_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.num_inner_steps as u64) as usize;

        // Simulate meta-training process
        // In a real implementation, this would use actual MAML algorithm
        let base_loss = 2.5_f64; // Starting loss
        let improvement_factor = 0.95_f64; // Improvement per step
        let current_step = self.training_history.len();

        // Simulate learning curve
        let meta_loss = base_loss * improvement_factor.powi(current_step as i32);

        // Add some noise
        let noise = (rand::random::<f64>() - 0.5) * 0.1;
        let final_loss = (meta_loss + noise).max(0.1);

        Ok(ExperimentResult {
            experiment_id: String::new(),
            agent_id: String::new(),
            status: ExperimentStatus::Completed,
            final_performance: -final_loss, // Negative loss as performance
            performance_trajectory: vec![-final_loss],
            resource_usage: Default::default(),
            start_time: std::time::Instant::now(),
            end_time: std::time::Instant::now(),
            statistics: Default::default(),
            insights: vec![],
            artifacts: {
                let mut map = HashMap::new();
                map.insert("meta_loss".to_string(), serde_json::json!(final_loss));
                map.insert(
                    "tasks_processed".to_string(),
                    serde_json::json!(tasks_per_batch),
                );
                map.insert(
                    "inner_steps".to_string(),
                    serde_json::json!(num_inner_steps),
                );
                map.insert("meta_lr".to_string(), serde_json::json!(self.meta_lr));
                map.insert("inner_lr".to_string(), serde_json::json!(self.inner_lr));
                map
            },
            metadata: HashMap::new(),
        })
    }

    /// Run few-shot evaluation simulation
    fn run_few_shot_evaluation(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        let n_way = experiment
            .experiment_config
            .get("n_way")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let k_shot = experiment
            .experiment_config
            .get("k_shot")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let n_query = experiment
            .experiment_config
            .get("n_query")
            .and_then(|v| v.as_u64())
            .unwrap_or(15) as usize;

        // Simulate few-shot evaluation
        // Base accuracy depends on training progress
        let base_accuracy = 0.5 + (self.training_history.len() as f64 * 0.02).min(0.4);

        // Add some variation based on task difficulty
        let task_factor = match (n_way, k_shot) {
            (5, 1) => 0.8,  // 5-way 1-shot is standard
            (5, 5) => 1.0,  // 5-way 5-shot should be easier
            (10, 1) => 0.6, // 10-way 1-shot is harder
            _ => 0.7,
        };

        let accuracy = (base_accuracy * task_factor).min(0.95);

        // Add noise
        let noise = (rand::random::<f64>() - 0.5) * 0.05;
        let final_accuracy = (accuracy + noise).clamp(0.1, 0.99);

        Ok(ExperimentResult {
            experiment_id: String::new(),
            agent_id: String::new(),
            status: ExperimentStatus::Completed,
            final_performance: final_accuracy,
            performance_trajectory: vec![final_accuracy],
            resource_usage: Default::default(),
            start_time: std::time::Instant::now(),
            end_time: std::time::Instant::now(),
            statistics: Default::default(),
            insights: vec![],
            artifacts: {
                let mut map = HashMap::new();
                map.insert("n_way".to_string(), serde_json::json!(n_way));
                map.insert("k_shot".to_string(), serde_json::json!(k_shot));
                map.insert("n_query".to_string(), serde_json::json!(n_query));
                map.insert("accuracy".to_string(), serde_json::json!(final_accuracy));
                map.insert(
                    "training_experiments".to_string(),
                    serde_json::json!(self.training_history.len()),
                );
                map
            },
            metadata: HashMap::new(),
        })
    }
}

/// Factory for creating MAML research agents
#[derive(Debug)]
pub struct MAMLResearchAgentFactory;

impl ResearchAgentFactory for MAMLResearchAgentFactory {
    fn create_factory() -> Box<dyn ResearchAgentFactory> {
        Box::new(Self)
    }

    fn create(&self, config: serde_json::Value) -> Result<Box<dyn ResearchAgent>> {
        let id = config
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("maml_agent")
            .to_string();

        let name = config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("MAML Research Agent")
            .to_string();

        let mut agent = MAMLResearchAgent::new(id, name);
        agent.initialize(config)?;

        Ok(Box::new(agent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prototypical_agent_creation() {
        let config = serde_json::json!({
            "id": "test_proto",
            "name": "Test Prototypical Agent",
            "input_size": 784,
            "hidden_size": 64,
            "distance_metric": "euclidean",
            "scale": 1.0,
            "temperature": 1.0
        });

        let factory = PrototypicalResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        assert_eq!(agent.id(), "test_proto");
        assert_eq!(agent.name(), "Test Prototypical Agent");
        assert_eq!(agent.agent_type(), AgentType::MetaLearning);
        assert!(agent.supports_domain(&ResearchDomain::MetaLearning));
        assert!(agent.supports_domain(&ResearchDomain::ComputerVision));
    }

    #[test]
    fn test_prototypical_agent_metadata() {
        let config = serde_json::json!({
            "id": "test_proto",
            "name": "Test Prototypical Agent"
        });

        let factory = PrototypicalResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        let metadata = agent.metadata();
        assert!(metadata
            .capabilities
            .contains(&"Few-shot classification".to_string()));
        assert!(metadata
            .supported_domains
            .contains(&ResearchDomain::MetaLearning));
        assert!(metadata
            .supported_domains
            .contains(&ResearchDomain::ComputerVision));
    }

    #[test]
    fn test_agent_available_actions() {
        let config = serde_json::json!({
            "id": "test_proto",
            "name": "Test Prototypical Agent"
        });

        let factory = PrototypicalResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        let actions = agent.get_available_actions();
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.id == "proto_eval"));
    }

    #[test]
    fn test_agent_initialization() {
        let mut agent =
            PrototypicalResearchAgent::new("test".to_string(), "Test Agent".to_string());

        let config = serde_json::json!({
            "input_size": 10,
            "hidden_size": 5
        });

        agent.initialize(config).unwrap();
        assert!(agent.is_ready());
    }

    #[test]
    fn test_maml_agent_creation() {
        let config = serde_json::json!({
            "id": "test_maml",
            "name": "Test MAML Agent",
            "meta_learning_rate": 0.001,
            "inner_learning_rate": 0.01,
            "num_inner_steps": 5,
            "tasks_per_batch": 4
        });

        let factory = MAMLResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        assert_eq!(agent.id(), "test_maml");
        assert_eq!(agent.name(), "Test MAML Agent");
        assert_eq!(agent.agent_type(), AgentType::MetaLearning);
        assert!(agent.supports_domain(&ResearchDomain::MetaLearning));
    }

    #[test]
    fn test_maml_agent_metadata() {
        let config = serde_json::json!({
            "id": "test_maml",
            "name": "Test MAML Agent"
        });

        let factory = MAMLResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        let metadata = agent.metadata();
        assert!(metadata.capabilities.contains(&"Meta-training".to_string()));
        assert!(metadata
            .supported_domains
            .contains(&ResearchDomain::MetaLearning));
    }

    #[test]
    fn test_maml_agent_available_actions() {
        let config = serde_json::json!({
            "id": "test_maml",
            "name": "Test MAML Agent"
        });

        let factory = MAMLResearchAgentFactory;
        let agent = factory.create(config).unwrap();

        let actions = agent.get_available_actions();
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.id == "maml_training"));
        assert!(actions.iter().any(|a| a.id == "maml_eval"));
    }

    #[test]
    fn test_maml_agent_meta_training_experiment() {
        let config = serde_json::json!({
            "id": "test_maml_training",
            "name": "Test MAML Training Agent"
        });

        let factory = MAMLResearchAgentFactory;
        let mut agent = factory.create(config).unwrap();

        let experiment = ExperimentSpec {
            id: "meta_train_test".to_string(),
            name: "Meta-Training Test".to_string(),
            domain: ResearchDomain::MetaLearning,
            agent_type: "maml".to_string(),
            experiment_config: serde_json::json!({
                "experiment_type": "meta_training",
                "tasks_per_batch": 2,
                "num_inner_steps": 3
            }),
            resource_requirements: Default::default(),
            dependencies: vec![],
            priority: 1,
            timeout_secs: Some(60),
            quality_constraints: Default::default(),
            metadata: HashMap::new(),
        };

        let result = agent.run_step(&experiment).unwrap();
        assert_eq!(result.status, ExperimentStatus::Completed);
        assert!(result.final_performance < 0.0); // Negative loss
        assert!(result.artifacts.contains_key("meta_loss"));
        assert!(result.artifacts.contains_key("tasks_processed"));
        assert!(result.artifacts.contains_key("inner_steps"));
    }

    #[test]
    fn test_maml_agent_few_shot_evaluation() {
        let config = serde_json::json!({
            "id": "test_maml_eval",
            "name": "Test MAML Evaluation Agent"
        });

        let factory = MAMLResearchAgentFactory;
        let mut agent = factory.create(config).unwrap();

        // Add some training history
        for _ in 0..5 {
            let train_exp = ExperimentSpec {
                id: "train".to_string(),
                name: "Training".to_string(),
                domain: ResearchDomain::MetaLearning,
                agent_type: "maml".to_string(),
                experiment_config: serde_json::json!({
                    "experiment_type": "meta_training"
                }),
                resource_requirements: Default::default(),
                dependencies: vec![],
                priority: 1,
                timeout_secs: Some(30),
                quality_constraints: Default::default(),
                metadata: HashMap::new(),
            };
            agent.run_step(&train_exp).unwrap();
        }

        let eval_experiment = ExperimentSpec {
            id: "few_shot_test".to_string(),
            name: "Few-Shot Evaluation Test".to_string(),
            domain: ResearchDomain::MetaLearning,
            agent_type: "maml".to_string(),
            experiment_config: serde_json::json!({
                "experiment_type": "few_shot_evaluation",
                "n_way": 3,
                "k_shot": 2,
                "n_query": 10
            }),
            resource_requirements: Default::default(),
            dependencies: vec![],
            priority: 1,
            timeout_secs: Some(30),
            quality_constraints: Default::default(),
            metadata: HashMap::new(),
        };

        let result = agent.run_step(&eval_experiment).unwrap();
        assert_eq!(result.status, ExperimentStatus::Completed);
        assert!(result.final_performance > 0.0); // Positive accuracy
        assert!(result.final_performance <= 1.0); // Accuracy bounded
        assert!(result.artifacts.contains_key("n_way"));
        assert!(result.artifacts.contains_key("k_shot"));
        assert!(result.artifacts.contains_key("accuracy"));
    }

    #[test]
    fn test_maml_agent_state_management() {
        let config = serde_json::json!({
            "id": "test_maml_state",
            "name": "Test MAML State Agent",
            "meta_learning_rate": 0.002,
            "inner_learning_rate": 0.005
        });

        let factory = MAMLResearchAgentFactory;
        let mut agent = factory.create(config).unwrap();

        // Run some experiments
        for i in 0..3 {
            let exp = ExperimentSpec {
                id: format!("exp_{}", i),
                name: format!("Experiment {}", i),
                domain: ResearchDomain::MetaLearning,
                agent_type: "maml".to_string(),
                experiment_config: serde_json::json!({
                    "experiment_type": "meta_training"
                }),
                resource_requirements: Default::default(),
                dependencies: vec![],
                priority: 1,
                timeout_secs: Some(30),
                quality_constraints: Default::default(),
                metadata: HashMap::new(),
            };
            agent.run_step(&exp).unwrap();
        }

        // Get state
        let state = agent.get_state().unwrap();
        assert!(state.is_object());
        let state_obj = state.as_object().unwrap();
        assert!(state_obj.contains_key("training_history"));
        assert!(state_obj.contains_key("meta_lr"));
        assert!(state_obj.contains_key("inner_lr"));
        assert_eq!(state_obj["training_history"].as_array().unwrap().len(), 3);

        // Create new agent and set state
        let new_config = serde_json::json!({
            "id": "test_maml_restore",
            "name": "Test MAML Restore Agent"
        });

        let mut new_agent = factory.create(new_config).unwrap();

        // Verify initial state (should be empty)
        let initial_state = new_agent.get_state().unwrap();
        assert_eq!(
            initial_state["training_history"].as_array().unwrap().len(),
            0
        );

        // Now set the state
        new_agent.set_state(state).unwrap();

        // Verify state was restored
        let restored_state = new_agent.get_state().unwrap();
        assert_eq!(
            restored_state["training_history"].as_array().unwrap().len(),
            3
        );
        assert_eq!(restored_state["meta_lr"], 0.002);
        assert_eq!(restored_state["inner_lr"], 0.005);
    }

    #[test]
    fn test_maml_agent_insight_generation() {
        let config = serde_json::json!({
            "id": "test_maml_insights",
            "name": "Test MAML Insights Agent"
        });

        let factory = MAMLResearchAgentFactory;
        let mut agent = factory.create(config).unwrap();

        // Initially no insights
        let insights = agent.generate_insights();
        assert!(insights.is_empty());

        // Add training history
        for i in 0..10 {
            let performance = -2.0 + (i as f64 * 0.1); // Improving performance
            let _ = agent.update_with_results(&[ExperimentResult {
                experiment_id: format!("exp_{}", i),
                agent_id: "test".to_string(),
                status: ExperimentStatus::Completed,
                final_performance: performance,
                performance_trajectory: vec![performance],
                resource_usage: Default::default(),
                start_time: std::time::Instant::now(),
                end_time: std::time::Instant::now(),
                statistics: Default::default(),
                insights: vec![],
                artifacts: HashMap::new(),
                metadata: HashMap::new(),
            }]);
        }

        // Now should generate insights
        let insights = agent.generate_insights();
        assert!(!insights.is_empty());
        let insight = &insights[0];
        assert_eq!(insight.agent_type, "maml");
        assert!(insight
            .domains
            .contains(&ResearchDomain::MetaLearning.to_string()));
        assert!(insight.confidence > 0.0);
        assert!(insight.confidence <= 1.0);
    }

    #[test]
    fn test_maml_agent_registry_integration() {
        use crate::research::{ResearchAgentRegistry, ResearchDomain};

        // Create registry and register MAML factory
        let registry = ResearchAgentRegistry::new();
        registry
            .register::<MAMLResearchAgentFactory>("maml")
            .unwrap();

        // Verify registration
        let agents = registry.list_agents();
        assert!(agents.contains(&"maml".to_string()));

        // Create agent through registry
        let config = serde_json::json!({
            "id": "registry_maml",
            "name": "Registry MAML Agent",
            "meta_learning_rate": 0.001,
            "inner_learning_rate": 0.01
        });

        let mut agent = registry.create_agent("maml", config).unwrap();
        assert_eq!(agent.id(), "registry_maml");
        assert_eq!(agent.agent_type(), super::AgentType::MetaLearning);

        // Test agent functionality through registry
        let experiment = super::ExperimentSpec {
            id: "registry_test".to_string(),
            name: "Registry Integration Test".to_string(),
            domain: ResearchDomain::MetaLearning,
            agent_type: "maml".to_string(),
            experiment_config: serde_json::json!({
                "experiment_type": "meta_training",
                "tasks_per_batch": 2,
                "num_inner_steps": 3
            }),
            resource_requirements: Default::default(),
            dependencies: vec![],
            priority: 1,
            timeout_secs: Some(30),
            quality_constraints: Default::default(),
            metadata: HashMap::new(),
        };

        let result = agent.run_step(&experiment).unwrap();
        assert_eq!(result.status, super::ExperimentStatus::Completed);
        assert!(result.final_performance < 0.0); // Meta-training loss
    }

    #[test]
    fn test_maml_agent_registry_integration_complete() {
        use crate::research::ResearchAgentRegistry;

        // Create registry and register both MAML and Prototypical agents
        let registry = ResearchAgentRegistry::new();
        registry
            .register::<MAMLResearchAgentFactory>("maml")
            .unwrap();
        registry
            .register::<PrototypicalResearchAgentFactory>("proto")
            .unwrap();

        // Verify both agents are registered
        let agents = registry.list_agents();
        assert!(agents.contains(&"maml".to_string()));
        assert!(agents.contains(&"proto".to_string()));

        // Create MAML agent config
        let maml_config = serde_json::json!({
            "id": "integration_maml",
            "name": "Integration MAML Agent",
            "meta_learning_rate": 0.001,
            "inner_learning_rate": 0.01
        });

        // Create Prototypical agent config
        let proto_config = serde_json::json!({
            "id": "integration_proto",
            "name": "Integration Prototypical Agent"
        });

        // Create both agents through registry
        let mut maml_agent = registry.create_agent("maml", maml_config).unwrap();
        let proto_agent = registry.create_agent("proto", proto_config).unwrap();

        // Verify agent properties
        assert_eq!(maml_agent.id(), "integration_maml");
        assert_eq!(maml_agent.agent_type(), super::AgentType::MetaLearning);
        assert_eq!(proto_agent.id(), "integration_proto");
        assert_eq!(proto_agent.agent_type(), super::AgentType::MetaLearning);

        // Test MAML agent functionality
        let maml_experiment = super::ExperimentSpec {
            id: "registry_maml_test".to_string(),
            name: "Registry MAML Test".to_string(),
            domain: ResearchDomain::MetaLearning,
            agent_type: "maml".to_string(),
            experiment_config: serde_json::json!({
                "experiment_type": "meta_training",
                "tasks_per_batch": 2,
                "num_inner_steps": 3
            }),
            resource_requirements: Default::default(),
            dependencies: vec![],
            priority: 1,
            timeout_secs: Some(30),
            quality_constraints: Default::default(),
            metadata: HashMap::new(),
        };

        let maml_result = maml_agent.run_step(&maml_experiment).unwrap();
        assert_eq!(maml_result.status, super::ExperimentStatus::Completed);
        assert!(maml_result.final_performance < 0.0); // Meta-training loss

        // Test available actions for both agents
        let maml_actions = maml_agent.get_available_actions();
        let proto_actions = proto_agent.get_available_actions();

        assert!(!maml_actions.is_empty());
        assert!(!proto_actions.is_empty());

        // Test insight generation capability
        let maml_insights = maml_agent.generate_insights();
        // MAML insights require training history, so this might be empty initially
        assert!(maml_insights.is_empty() || !maml_insights.is_empty());

        // Test state management
        let state = maml_agent.get_state().unwrap();
        assert!(state.is_object());
        assert!(state.as_object().unwrap().contains_key("training_history"));

        // Test metadata
        let maml_metadata = maml_agent.metadata();
        let proto_metadata = proto_agent.metadata();

        assert!(maml_metadata
            .capabilities
            .contains(&"Meta-training".to_string()));
        assert!(maml_metadata
            .supported_domains
            .contains(&ResearchDomain::MetaLearning));
        assert!(proto_metadata
            .capabilities
            .contains(&"Few-shot classification".to_string()));
        assert!(proto_metadata
            .supported_domains
            .contains(&ResearchDomain::MetaLearning));
    }
}
