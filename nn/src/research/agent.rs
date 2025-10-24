//! Unified Research Agent Traits
//!
//! This module defines common traits and interfaces for research agents
//! across NAS, HPO, and meta-learning systems.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{NNError, Result};

use super::{ExperimentResult, ExperimentSpec, ExperimentStatus, ResearchDomain, ResearchInsight};

/// Core trait for research agents
pub trait ResearchAgent: Send + Sync {
    /// Get agent identifier
    fn id(&self) -> &str;

    /// Get agent name
    fn name(&self) -> &str;

    /// Get agent type (NAS, HPO, Meta)
    fn agent_type(&self) -> AgentType;

    /// Get metadata about the agent
    fn metadata(&self) -> AgentMetadata;

    /// Check if agent supports given domain
    fn supports_domain(&self, domain: &ResearchDomain) -> bool;

    /// Initialize agent with configuration
    fn initialize(&mut self, config: serde_json::Value) -> Result<()>;

    /// Run a single experiment step
    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult>;

    /// Get available actions/experiments for current state
    fn get_available_actions(&self) -> Vec<ExperimentSpec>;

    /// Update agent with experiment results
    fn update_with_results(&mut self, results: &[ExperimentResult]) -> Result<()>;

    /// Get current best configuration/result
    fn get_best_result(&self) -> Option<ExperimentResult>;

    /// Get agent's current state as serializable data
    fn get_state(&self) -> Result<serde_json::Value>;

    /// Set agent state from serialized data
    fn set_state(&mut self, state: serde_json::Value) -> Result<()>;

    /// Check if agent is ready to run experiments
    fn is_ready(&self) -> bool;

    /// Get resource requirements for next experiment
    fn get_resource_requirements(&self) -> ResourceRequirements;

    /// Generate insights for knowledge transfer
    fn generate_insights(&self) -> Vec<ResearchInsight>;
}

/// Factory trait for creating research agents
pub trait ResearchAgentFactory {
    /// Create a new instance of the agent
    fn create(&self, config: serde_json::Value) -> Result<Box<dyn ResearchAgent>>;

    /// Create a factory instance
    fn create_factory() -> Box<dyn ResearchAgentFactory> where Self: Sized;
}

/// Agent type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// Neural Architecture Search
    NAS,
    /// Hyperparameter Optimization
    HPO,
    /// Meta-Learning
    MetaLearning,
    /// Hybrid/Multi-agent
    Hybrid,
}

/// Agent metadata
#[derive(Debug, Clone)]
pub struct AgentMetadata {
    /// Agent version
    pub version: String,
    /// Supported research domains
    pub supported_domains: Vec<ResearchDomain>,
    /// Expected resource requirements
    pub resource_profile: ResourceProfile,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
    /// Capabilities
    pub capabilities: Vec<String>,
}

impl Default for AgentMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            supported_domains: vec![ResearchDomain::GeneralML],
            resource_profile: ResourceProfile::default(),
            performance_characteristics: PerformanceCharacteristics::default(),
            capabilities: Vec::new(),
        }
    }
}

/// Resource requirements for agent experiments
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: usize,
    /// GPU memory required (GB)
    pub gpu_memory_gb: f64,
    /// System memory required (GB)
    pub system_memory_gb: f64,
    /// Storage required (GB)
    pub storage_gb: f64,
    /// Estimated execution time (seconds)
    pub estimated_time_secs: u64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            gpu_memory_gb: 1.0,
            system_memory_gb: 2.0,
            storage_gb: 1.0,
            estimated_time_secs: 60,
        }
    }
}

/// Resource profile for agent type
#[derive(Debug, Clone)]
pub struct ResourceProfile {
    /// Minimum CPU cores
    pub min_cpu_cores: usize,
    /// Maximum CPU cores
    pub max_cpu_cores: usize,
    /// Typical GPU memory usage (GB)
    pub typical_gpu_memory_gb: f64,
    /// Typical system memory usage (GB)
    pub typical_system_memory_gb: f64,
    /// Typical storage usage (GB)
    pub typical_storage_gb: f64,
    /// Scalability factor
    pub scalability: ScalabilityProfile,
}

impl Default for ResourceProfile {
    fn default() -> Self {
        Self {
            min_cpu_cores: 1,
            max_cpu_cores: 8,
            typical_gpu_memory_gb: 4.0,
            typical_system_memory_gb: 8.0,
            typical_storage_gb: 10.0,
            scalability: ScalabilityProfile::default(),
        }
    }
}

/// Scalability profile
#[derive(Debug, Clone)]
pub struct ScalabilityProfile {
    /// Can scale with more CPU cores
    pub scales_with_cpu: bool,
    /// Can scale with more GPU memory
    pub scales_with_gpu_memory: bool,
    /// Can benefit from distributed execution
    pub supports_distributed: bool,
    /// Parallel efficiency factor (0.0 to 1.0)
    pub parallel_efficiency: f64,
}

impl Default for ScalabilityProfile {
    fn default() -> Self {
        Self {
            scales_with_cpu: true,
            scales_with_gpu_memory: true,
            supports_distributed: false,
            parallel_efficiency: 0.7,
        }
    }
}

/// Performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Expected convergence speed (iterations per improvement)
    pub convergence_speed: f64,
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
    /// Exploration vs exploitation balance
    pub exploration_factor: f64,
    /// Adaptability to new domains
    pub adaptability: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
}

impl Default for PerformanceCharacteristics {
    fn default() -> Self {
        Self {
            convergence_speed: 10.0,
            reliability: 0.8,
            exploration_factor: 0.5,
            adaptability: 0.6,
            computational_efficiency: 0.7,
        }
    }
}

/// Multi-agent coordinator for combining multiple research agents
pub struct MultiAgentCoordinator {
    /// Registered agents
    agents: HashMap<String, Box<dyn ResearchAgent>>,
    /// Coordination strategy
    strategy: CoordinationStrategy,
    /// Shared knowledge base
    knowledge_base: Arc<RwLock<SharedKnowledgeBase>>,
    /// Communication channels between agents
    communication_channels: HashMap<(String, String), CommunicationChannel>,
}

impl MultiAgentCoordinator {
    /// Create new multi-agent coordinator
    pub fn new(strategy: CoordinationStrategy) -> Self {
        Self {
            agents: HashMap::new(),
            strategy,
            knowledge_base: Arc::new(RwLock::new(SharedKnowledgeBase::new())),
            communication_channels: HashMap::new(),
        }
    }

    /// Register an agent
    pub fn register_agent(&mut self, id: String, agent: Box<dyn ResearchAgent>) -> Result<()> {
        if self.agents.contains_key(&id) {
            return Err(NNError::InvalidConfiguration {
                message: format!("Agent with id '{}' already registered", id),
            });
        }

        self.agents.insert(id.clone(), agent);
        self.initialize_communication_channels(id)?;

        Ok(())
    }

    /// Initialize communication channels for new agent
    fn initialize_communication_channels(&mut self, new_agent_id: String) -> Result<()> {
        for existing_id in self.agents.keys() {
            if existing_id != &new_agent_id {
                let channel = CommunicationChannel::new(existing_id.clone(), new_agent_id.clone());
                self.communication_channels.insert(
                    (existing_id.clone(), new_agent_id.clone()),
                    channel,
                );
            }
        }
        Ok(())
    }

    /// Execute coordinated research step
    pub fn execute_coordinated_step(&mut self, context: &ResearchContext) -> Result<CoordinatedResult> {
        match &self.strategy {
            CoordinationStrategy::Sequential => self.execute_sequential(context),
            CoordinationStrategy::Parallel => self.execute_parallel(context),
            CoordinationStrategy::Hierarchical => self.execute_hierarchical(context),
            CoordinationStrategy::Collaborative => self.execute_collaborative(context),
        }
    }

    /// Sequential execution strategy
    fn execute_sequential(&mut self, context: &ResearchContext) -> Result<CoordinatedResult> {
        let mut results = Vec::new();

        // First pass: collect insights for each agent
        let insights_map: HashMap<String, Vec<ResearchInsight>> = self.agents
            .iter()
            .map(|(agent_id, agent)| {
                let insights = self.get_relevant_insights(&**agent).unwrap_or_default();
                (agent_id.clone(), insights)
            })
            .collect();

        // Second pass: execute agents with their insights
        let mut execution_results = Vec::new();

        for (agent_id, agent) in &mut self.agents {
            let relevant_insights = insights_map.get(agent_id).cloned().unwrap_or_default();

            // Update agent with insights
            if !relevant_insights.is_empty() {
                let insights_results = relevant_insights
                    .into_iter()
                    .map(|insight| {
                        // Convert insight to experiment result format
                        ExperimentResult {
                            experiment_id: format!("insight_{}", insight.id),
                            agent_id: agent_id.clone(),
                            status: ExperimentStatus::Completed,
                            final_performance: insight.performance_impact,
                            performance_trajectory: vec![insight.performance_impact],
                            resource_usage: Default::default(),
                            start_time: insight.timestamp,
                            end_time: insight.timestamp,
                            statistics: Default::default(),
                            insights: vec![insight],
                            artifacts: HashMap::new(),
                            metadata: HashMap::new(),
                        }
                    })
                    .collect::<Vec<_>>();

                agent.update_with_results(&insights_results)?;
            }

            // Execute agent step
            let available_actions = agent.get_available_actions();
            if let Some(experiment) = available_actions.first() {
                let result = agent.run_step(experiment)?;
                execution_results.push((agent_id.clone(), result));
            }
        }

        // Third pass: share insights (no mutable borrow conflict)
        for (agent_id, result) in &execution_results {
            results.push(result.clone());
            self.share_insights(result)?;
        }

        Ok(CoordinatedResult {
            individual_results: results,
            coordinated_performance: self.evaluate_coordination(),
            knowledge_transfer_score: self.evaluate_knowledge_transfer(),
        })
    }

    /// Parallel execution strategy
    fn execute_parallel(&mut self, _context: &ResearchContext) -> Result<CoordinatedResult> {
        // Parallel execution would require async runtime
        // For now, delegate to sequential
        self.execute_sequential(_context)
    }

    /// Hierarchical execution strategy
    fn execute_hierarchical(&mut self, context: &ResearchContext) -> Result<CoordinatedResult> {
        // Identify master agent (first registered)
        let master_id = self.agents.keys().next().cloned()
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: "No agents registered for hierarchical coordination".to_string(),
            })?;

        // Master agent makes decisions
        let master_exp = {
            let master_agent = self.agents.get_mut(&master_id).unwrap();
            master_agent.get_available_actions().first().cloned()
        };

        if let Some(experiment) = master_exp {
            let mut results = Vec::new();

            // All agents work on the master's experiment
            for (agent_id, agent) in &mut self.agents {
                let result = agent.run_step(&experiment)?;
                results.push(result);
            }

            Ok(CoordinatedResult {
                individual_results: results,
                coordinated_performance: self.evaluate_coordination(),
                knowledge_transfer_score: self.evaluate_knowledge_transfer(),
            })
        } else {
            self.execute_sequential(context)
        }
    }

    /// Collaborative execution strategy
    fn execute_collaborative(&mut self, context: &ResearchContext) -> Result<CoordinatedResult> {
        // Collaborative execution allows agents to share intermediate results
        let mut results = Vec::new();

        // First pass: collect collaborative inputs for each agent
        let collaborative_inputs: HashMap<String, CollaborativeInput> = self.agents
            .keys()
            .map(|agent_id| {
                let input = self.get_collaborative_input(agent_id).unwrap_or_default();
                (agent_id.clone(), input)
            })
            .collect();

        // Second pass: execute agents with their collaborative inputs
        let mut collab_results = Vec::new();

        for (agent_id, agent) in &mut self.agents {
            // Execute with collaborative input
            let available_actions = agent.get_available_actions();
            if let Some(experiment) = available_actions.first() {
                let result = agent.run_step(experiment)?;
                collab_results.push((agent_id.clone(), result));
            }
        }

        // Third pass: share intermediate results (no borrow conflict)
        for (agent_id, result) in &collab_results {
            results.push(result.clone());
            self.share_intermediate_results(agent_id, result)?;
        }

        Ok(CoordinatedResult {
            individual_results: results,
            coordinated_performance: self.evaluate_coordination(),
            knowledge_transfer_score: self.evaluate_knowledge_transfer(),
        })
    }

    /// Get relevant insights for an agent
    fn get_relevant_insights(&self, agent: &dyn ResearchAgent) -> Result<Vec<ResearchInsight>> {
        let knowledge_base = self.knowledge_base.read().unwrap();
        let agent_domains = agent.metadata().supported_domains;

        Ok(knowledge_base.insights.iter()
            .filter(|insight| {
                insight.agent_type != agent.id() && // Don't give insights to self
                insight.domains.iter().any(|domain| agent_domains.contains(domain))
            })
            .cloned()
            .collect())
    }

    /// Share insights from experiment result
    fn share_insights(&self, result: &ExperimentResult) -> Result<()> {
        let mut knowledge_base = self.knowledge_base.write().unwrap();
        knowledge_base.insights.extend(result.insights.clone());
        Ok(())
    }

    /// Get collaborative input from other agents
    fn get_collaborative_input(&self, agent_id: &str) -> Result<CollaborativeInput> {
        // Simplified collaborative input generation
        Ok(CollaborativeInput {
            suggestions: Vec::new(),
            constraints: Vec::new(),
            shared_knowledge: serde_json::Value::Null,
        })
    }

    /// Share intermediate results with other agents
    fn share_intermediate_results(&self, agent_id: &str, result: &ExperimentResult) -> Result<()> {
        // Store intermediate results for collaborative use
        Ok(())
    }

    /// Evaluate coordination performance
    fn evaluate_coordination(&self) -> f64 {
        // Simplified coordination evaluation
        0.8
    }

    /// Evaluate knowledge transfer effectiveness
    fn evaluate_knowledge_transfer(&self) -> f64 {
        let knowledge_base = self.knowledge_base.read().unwrap();
        knowledge_base.insights.len() as f64 * 0.1
    }
}

/// Coordination strategy
#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    /// Execute agents sequentially
    Sequential,
    /// Execute agents in parallel
    Parallel,
    /// Hierarchical coordination with master agent
    Hierarchical,
    /// Collaborative execution with shared knowledge
    Collaborative,
}

/// Research context for coordination
#[derive(Debug, Clone)]
pub struct ResearchContext {
    /// Current research goal
    pub goal: String,
    /// Available resources
    pub resources: ResourceRequirements,
    /// Time constraints
    pub time_budget_secs: u64,
    /// Quality requirements
    pub quality_threshold: f64,
}

/// Coordinated execution result
#[derive(Debug, Clone)]
pub struct CoordinatedResult {
    /// Individual agent results
    pub individual_results: Vec<ExperimentResult>,
    /// Coordinated performance score
    pub coordinated_performance: f64,
    /// Knowledge transfer effectiveness
    pub knowledge_transfer_score: f64,
}

/// Communication channel between agents
#[derive(Debug, Clone)]
pub struct CommunicationChannel {
    /// Source agent ID
    pub source_id: String,
    /// Target agent ID
    pub target_id: String,
    /// Message queue
    pub messages: Vec<AgentMessage>,
}

impl CommunicationChannel {
    /// Create new communication channel
    pub fn new(source_id: String, target_id: String) -> Self {
        Self {
            source_id,
            target_id,
            messages: Vec::new(),
        }
    }

    /// Send message
    pub fn send(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    /// Receive messages
    pub fn receive(&mut self) -> Vec<AgentMessage> {
        std::mem::take(&mut self.messages)
    }
}

/// Message between agents
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Message type
    pub message_type: MessageType,
    /// Content
    pub content: serde_json::Value,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Message type
#[derive(Debug, Clone)]
pub enum MessageType {
    /// Insight sharing
    Insight,
    /// Request for collaboration
    CollaborationRequest,
    /// Response to collaboration
    CollaborationResponse,
    /// Resource request
    ResourceRequest,
    /// Status update
    StatusUpdate,
}

/// Shared knowledge base for multi-agent coordination
#[derive(Debug, Clone)]
pub struct SharedKnowledgeBase {
    /// Stored insights
    pub insights: Vec<ResearchInsight>,
    /// Agent performance history
    pub performance_history: HashMap<String, Vec<f64>>,
    /// Successful patterns
    pub successful_patterns: Vec<Pattern>,
}

impl SharedKnowledgeBase {
    /// Create new knowledge base
    pub fn new() -> Self {
        Self {
            insights: Vec::new(),
            performance_history: HashMap::new(),
            successful_patterns: Vec::new(),
        }
    }
}

/// Learned pattern from successful research
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern description
    pub description: String,
    /// Applicable domains
    pub domains: Vec<ResearchDomain>,
    /// Success rate
    pub success_rate: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Collaborative input for agent coordination
#[derive(Debug, Clone, Default)]
pub struct CollaborativeInput {
    /// Suggestions from other agents
    pub suggestions: Vec<AgentSuggestion>,
    /// Constraints to consider
    pub constraints: Vec<Constraint>,
    /// Shared knowledge data
    pub shared_knowledge: serde_json::Value,
}

/// Suggestion from another agent
#[derive(Debug, Clone)]
pub struct AgentSuggestion {
    /// Suggesting agent ID
    pub agent_id: String,
    /// Suggestion type
    pub suggestion_type: String,
    /// Suggestion data
    pub data: serde_json::Value,
    /// Confidence level
    pub confidence: f64,
}

/// Constraint for agent behavior
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Priority level
    pub priority: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Mock research agent for testing
    struct MockResearchAgent {
        id: String,
        agent_type: AgentType,
        step_count: usize,
        best_performance: f64,
    }

    impl MockResearchAgent {
        fn new(id: &str, agent_type: AgentType) -> Self {
            Self {
                id: id.to_string(),
                agent_type,
                step_count: 0,
                best_performance: 0.0,
            }
        }
    }

    impl ResearchAgent for MockResearchAgent {
        fn id(&self) -> &str { &self.id }
        fn name(&self) -> &str { &self.id }
        fn agent_type(&self) -> AgentType { self.agent_type.clone() }

        fn metadata(&self) -> AgentMetadata {
            AgentMetadata {
                version: "1.0.0".to_string(),
                supported_domains: vec![ResearchDomain::GeneralML],
                resource_profile: ResourceProfile::default(),
                performance_characteristics: PerformanceCharacteristics::default(),
                capabilities: vec!["test".to_string()],
            }
        }

        fn supports_domain(&self, _domain: &ResearchDomain) -> bool { true }
        fn initialize(&mut self, _config: serde_json::Value) -> Result<()> { Ok(()) }

        fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
            self.step_count += 1;
            self.best_performance += 0.1;

            Ok(ExperimentResult {
                experiment_id: experiment.id.clone(),
                agent_id: self.id.clone(),
                status: ExperimentStatus::Completed,
                final_performance: self.best_performance,
                performance_trajectory: vec![self.best_performance],
                resource_usage: Default::default(),
                start_time: std::time::Instant::now(),
                end_time: std::time::Instant::now(),
                statistics: Default::default(),
                insights: Vec::new(),
                artifacts: HashMap::new(),
                metadata: HashMap::new(),
            })
        }

        fn get_available_actions(&self) -> Vec<ExperimentSpec> {
            vec![ExperimentSpec::new(
                format!("exp_{}", self.id),
                format!("Research Experiment {}", self.id),
                super::ResearchDomain::GeneralML,
                "dummy_agent".to_string(),
            )]
        }

        fn update_with_results(&mut self, _results: &[ExperimentResult]) -> Result<()> { Ok(()) }
        fn get_best_result(&self) -> Option<ExperimentResult> { None }
        fn get_state(&self) -> Result<serde_json::Value> { Ok(json!({"step_count": self.step_count})) }
        fn set_state(&mut self, state: serde_json::Value) -> Result<()> {
            self.step_count = state["step_count"].as_u64().unwrap_or(0) as usize;
            Ok(())
        }

        fn is_ready(&self) -> bool { true }
        fn get_resource_requirements(&self) -> ResourceRequirements { ResourceRequirements::default() }
        fn generate_insights(&self) -> Vec<ResearchInsight> { Vec::new() }
    }

    #[test]
    fn test_agent_metadata() {
        let metadata = AgentMetadata::default();
        assert_eq!(metadata.version, "1.0.0");
        assert!(!metadata.supported_domains.is_empty());
    }

    #[test]
    fn test_multi_agent_coordinator() {
        let mut coordinator = MultiAgentCoordinator::new(CoordinationStrategy::Sequential);

        // Register mock agents
        let agent1 = Box::new(MockResearchAgent::new("agent1", AgentType::HPO));
        let agent2 = Box::new(MockResearchAgent::new("agent2", AgentType::NAS));

        coordinator.register_agent("agent1".to_string(), agent1).unwrap();
        coordinator.register_agent("agent2".to_string(), agent2).unwrap();

        // Execute coordinated step
        let context = ResearchContext {
            goal: "Test coordination".to_string(),
            resources: ResourceRequirements::default(),
            time_budget_secs: 3600,
            quality_threshold: 0.8,
        };

        let result = coordinator.execute_coordinated_step(&context).unwrap();
        assert!(!result.individual_results.is_empty());
    }
}
