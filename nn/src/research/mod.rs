//! Unified Research Framework
//!
//! This module consolidates research agents across NAS, HPO, and meta-learning
//! into a unified framework with shared abstractions, experiment orchestration,
//! and knowledge transfer capabilities.

use std::sync::{Arc, RwLock};
use std::time::Instant;

// Re-export all types publicly

pub mod agent;
pub mod experiment;
pub mod orchestrator;
pub mod registry;
pub mod workflow;

// Re-export unified research framework types
pub use agent::{ResearchAgent, ResearchAgentFactory, AgentMetadata};
pub use experiment::{ExperimentSpec, ExperimentResult, ExperimentStatus};
pub use orchestrator::ResearchOrchestrator;
pub use registry::ResearchAgentRegistry;
pub use workflow::{ResearchWorkflow, WorkflowTemplate};

/// Unified research framework configuration
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// Maximum concurrent experiments
    pub max_concurrent_experiments: usize,
    /// Resource allocation limits
    pub resource_limits: ResourceLimits,
    /// Default experiment timeout (seconds)
    pub experiment_timeout_secs: u64,
    /// Knowledge transfer enabled
    pub knowledge_transfer_enabled: bool,
    /// Cross-validation settings
    pub cross_validation_folds: Option<usize>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_experiments: 4,
            resource_limits: ResourceLimits::default(),
            experiment_timeout_secs: 3600, // 1 hour
            knowledge_transfer_enabled: true,
            cross_validation_folds: Some(5),
        }
    }
}

/// Resource allocation limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum GPU memory (GB)
    pub max_gpu_memory_gb: f64,
    /// Maximum CPU cores
    pub max_cpu_cores: usize,
    /// Maximum memory (GB)
    pub max_memory_gb: f64,
    /// Maximum disk space (GB)
    pub max_disk_gb: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_gpu_memory_gb: 16.0,
            max_cpu_cores: 8,
            max_memory_gb: 32.0,
            max_disk_gb: 100.0,
        }
    }
}

/// Research domain enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResearchDomain {
    /// Computer vision tasks
    ComputerVision,
    /// Natural language processing
    NLP,
    /// Reinforcement learning
    ReinforcementLearning,
    /// General machine learning
    GeneralML,
    /// Meta-learning and few-shot learning
    MetaLearning,
    /// Automated machine learning
    AutoML,
}

/// Unified research metrics
#[derive(Debug, Clone)]
pub struct ResearchMetrics {
    /// Total experiments conducted
    pub total_experiments: usize,
    /// Successful experiments
    pub successful_experiments: usize,
    /// Failed experiments
    pub failed_experiments: usize,
    /// Average experiment duration
    pub avg_experiment_duration: f64,
    /// Best performance achieved
    pub best_performance: f64,
    /// Research efficiency score
    pub research_efficiency: f64,
    /// Knowledge transfer effectiveness
    pub knowledge_transfer_score: f64,
}

impl Default for ResearchMetrics {
    fn default() -> Self {
        Self {
            total_experiments: 0,
            successful_experiments: 0,
            failed_experiments: 0,
            avg_experiment_duration: 0.0,
            best_performance: f64::NEG_INFINITY,
            research_efficiency: 0.0,
            knowledge_transfer_score: 0.0,
        }
    }
}

/// Research insight for knowledge transfer
#[derive(Debug, Clone)]
pub struct ResearchInsight {
    /// Insight identifier
    pub id: String,
    /// Agent that generated the insight
    pub agent_type: String,
    /// Applicable research domains
    pub domains: Vec<ResearchDomain>,
    /// Performance impact score
    pub performance_impact: f64,
    /// Confidence level
    pub confidence: f64,
    /// Transferable knowledge data
    pub knowledge_data: serde_json::Value,
    /// Timestamp
    pub timestamp: Instant,
}

/// Comprehensive research framework
pub struct UnifiedResearchFramework {
    /// Framework configuration
    pub config: ResearchConfig,
    /// Research agent registry
    pub registry: ResearchAgentRegistry,
    /// Experiment orchestrator
    pub orchestrator: ResearchOrchestrator,
    /// Research metrics
    pub metrics: Arc<RwLock<ResearchMetrics>>,
    /// Research insights for knowledge transfer
    pub insights: Arc<RwLock<Vec<ResearchInsight>>>,
}

impl UnifiedResearchFramework {
    /// Create new unified research framework
    pub fn new(config: ResearchConfig) -> Self {
        Self {
            config: config.clone(),
            registry: ResearchAgentRegistry::new(),
            orchestrator: ResearchOrchestrator::new(config),
            metrics: Arc::new(RwLock::new(ResearchMetrics::default())),
            insights: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a research agent
    pub fn register_agent<A: ResearchAgent + ResearchAgentFactory + 'static>(
        &mut self,
        agent_type: &str,
    ) -> crate::error::Result<()> {
        self.registry.register::<A>(agent_type)
    }

    /// Create research agent instance
    pub fn create_agent(&self, agent_type: &str, config: serde_json::Value) -> crate::error::Result<Box<dyn ResearchAgent>> {
        self.registry.create_agent(agent_type, config)
    }

    /// Execute research workflow
    pub fn execute_workflow(&mut self, workflow: &ResearchWorkflow) -> crate::error::Result<ExperimentResult> {
        self.orchestrator.execute_workflow(workflow, &self.registry)
    }

    /// Add research insight for knowledge transfer
    pub fn add_insight(&self, insight: ResearchInsight) {
        let mut insights = self.insights.write().unwrap();
        insights.push(insight);
    }

    /// Get applicable insights for domain
    pub fn get_domain_insights(&self, domain: &ResearchDomain) -> Vec<ResearchInsight> {
        let insights = self.insights.read().unwrap();
        insights
            .iter()
            .filter(|insight| insight.domains.contains(domain))
            .cloned()
            .collect()
    }

    /// Update research metrics
    pub fn update_metrics(&self, result: &ExperimentResult) {
        let mut metrics = self.metrics.write().unwrap();

        metrics.total_experiments += 1;

        match result.status {
            ExperimentStatus::Completed => {
                metrics.successful_experiments += 1;
                if result.final_performance > metrics.best_performance {
                    metrics.best_performance = result.final_performance;
                }
            }
            ExperimentStatus::Failed => {
                metrics.failed_experiments += 1;
            }
            _ => {}
        }

        // Update average duration
        let duration = result.end_time.duration_since(result.start_time).as_secs_f64();
        let total_duration = metrics.avg_experiment_duration * (metrics.total_experiments - 1) as f64;
        metrics.avg_experiment_duration = (total_duration + duration) / metrics.total_experiments as f64;
    }

    /// Get current research metrics
    pub fn get_metrics(&self) -> ResearchMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Generate research summary report
    pub fn generate_report(&self) -> ResearchReport {
        let metrics = self.get_metrics();
        let insights = self.insights.read().unwrap();

        ResearchReport {
            total_experiments: metrics.total_experiments,
            success_rate: metrics.successful_experiments as f64 / metrics.total_experiments as f64,
            avg_duration: metrics.avg_experiment_duration,
            best_performance: metrics.best_performance,
            total_insights: insights.len(),
            research_efficiency: metrics.research_efficiency,
        }
    }
}

/// Research framework report
#[derive(Debug, Clone)]
pub struct ResearchReport {
    /// Total experiments conducted
    pub total_experiments: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average experiment duration (seconds)
    pub avg_duration: f64,
    /// Best performance achieved
    pub best_performance: f64,
    /// Total insights generated
    pub total_insights: usize,
    /// Research efficiency score
    pub research_efficiency: f64,
}

impl std::fmt::Display for ResearchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Research Framework Report:\n\
             ├─ Total Experiments: {}\n\
             ├─ Success Rate: {:.1}%\n\
             ├─ Avg Duration: {:.1}s\n\
             ├─ Best Performance: {:.4}\n\
             ├─ Insights Generated: {}\n\
             └─ Research Efficiency: {:.3}",
            self.total_experiments,
            self.success_rate * 100.0,
            self.avg_duration,
            self.best_performance,
            self.total_insights,
            self.research_efficiency
        )
    }
}
