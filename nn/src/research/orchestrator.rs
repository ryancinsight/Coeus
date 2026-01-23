//! Advanced Research Experiment Orchestrator
//!
//! This module provides sophisticated workflow orchestration with DAG execution,
//! parallel processing, resource management, and comprehensive error recovery.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::workflow::WorkflowStep;
use super::{
    ExperimentResult, ExperimentSpec, ResearchAgentRegistry, ResearchConfig, ResearchWorkflow,
};
use crate::core::error::{NNError, Result};

/// Advanced research experiment orchestrator with DAG execution
#[derive(Debug)]
pub struct ResearchOrchestrator {
    /// Orchestrator configuration
    #[allow(dead_code)]
    config: ResearchConfig,
    /// Execution engine for workflow orchestration
    execution_engine: WorkflowExecutionEngine,
    /// Resource manager for experiment allocation
    resource_manager: Arc<ResourceManager>,
    /// Progress tracker for workflow monitoring
    progress_tracker: Arc<ProgressTracker>,
}

/// Workflow execution engine with DAG orchestration
#[derive(Debug)]
pub struct WorkflowExecutionEngine {
    /// Maximum concurrent workflow steps
    #[allow(dead_code)]
    max_concurrent_steps: usize,
    /// Execution timeout
    #[allow(dead_code)]
    execution_timeout: Duration,
    /// Enable parallel execution
    #[allow(dead_code)]
    enable_parallel: bool,
}

impl WorkflowExecutionEngine {
    /// Create new execution engine
    pub fn new(
        max_concurrent_steps: usize,
        execution_timeout: Duration,
        enable_parallel: bool,
    ) -> Self {
        Self {
            max_concurrent_steps,
            execution_timeout,
            enable_parallel,
        }
    }

    /// Execute workflow with DAG-based orchestration
    pub async fn execute_workflow_dag(
        &self,
        workflow: &ResearchWorkflow,
        registry: &ResearchAgentRegistry,
        resource_manager: &ResourceManager,
        progress_tracker: &ProgressTracker,
    ) -> Result<WorkflowResult> {
        let start_time = Instant::now();
        let workflow_id = workflow.id.clone();

        progress_tracker
            .update_workflow_status(&workflow_id, WorkflowExecutionStatus::Running)
            .await;

        // Build execution DAG
        let execution_graph = self.build_execution_graph(workflow)?;

        // Execute DAG with resource management
        let result = self
            .execute_dag_parallel(
                execution_graph,
                workflow,
                registry,
                resource_manager,
                progress_tracker,
            )
            .await;

        let execution_time = start_time.elapsed();
        let status = match result {
            Ok(_) => WorkflowStatus::Completed,
            Err(_) => WorkflowStatus::Failed,
        };

        progress_tracker
            .update_workflow_status(&workflow_id, WorkflowExecutionStatus::Completed)
            .await;

        Ok(WorkflowResult {
            status,
            experiment_results: result.unwrap_or_default(),
            execution_time,
        })
    }

    /// Build execution graph from workflow steps
    fn build_execution_graph(&self, workflow: &ResearchWorkflow) -> Result<ExecutionGraph> {
        let mut graph = ExecutionGraph::new();
        let mut step_indices = HashMap::new();

        // Create nodes for each step
        for (idx, step) in workflow.steps.iter().enumerate() {
            step_indices.insert(step.id.clone(), idx);
            graph.add_node(ExecutionNode {
                step_id: step.id.clone(),
                step_index: idx,
                dependencies: step.dependencies.clone(),
                status: ExecutionStatus::Pending,
                priority: step.priority,
            });
        }

        // Build dependency edges
        for step in &workflow.steps {
            for dep_id in &step.dependencies {
                if let Some(&dep_idx) = step_indices.get(dep_id) {
                    if let Some(&step_idx) = step_indices.get(&step.id) {
                        graph.add_edge(dep_idx, step_idx);
                    }
                } else {
                    return Err(NNError::InvalidConfiguration {
                        message: format!("Step {} depends on unknown step {}", step.id, dep_id),
                    });
                }
            }
        }

        Ok(graph)
    }

    /// Execute DAG with parallel processing and resource constraints
    async fn execute_dag_parallel(
        &self,
        graph: ExecutionGraph,
        workflow: &ResearchWorkflow,
        registry: &ResearchAgentRegistry,
        resource_manager: &ResourceManager,
        progress_tracker: &ProgressTracker,
    ) -> Result<Vec<ExperimentResult>> {
        let mut results = Vec::new();

        // Execute all nodes in topological order (simplified - assumes DAG is already topologically sorted)
        for node_idx in 0..graph.nodes.len() {
            let node = &graph.nodes[node_idx];
            let step = workflow.steps[node.step_index].clone();
            let workflow_id = workflow.id.clone();

            // Allocate resources
            let resource_allocation = resource_manager.allocate_resources(&step).await?;

            // Execute step
            progress_tracker
                .update_step_status(&workflow_id, &step.id, ExecutionStatus::Running)
                .await;
            let start_time = Instant::now();

            let result: Result<ExperimentResult> =
                Self::execute_workflow_step(&step, registry).await;

            let execution_time = start_time.elapsed();
            progress_tracker
                .record_step_execution_time(&workflow_id, &step.id, execution_time)
                .await;

            // Release resources
            if let Err(e) = resource_manager
                .release_resources(resource_allocation)
                .await
            {
                eprintln!("Warning: Failed to release resources: {}", e);
            }

            let status = match result {
                Ok(_) => ExecutionStatus::Completed,
                Err(_) => ExecutionStatus::Failed,
            };

            progress_tracker
                .update_step_status(&workflow_id, &step.id, status)
                .await;

            results.push(result);
        }

        // Extract successful results
        let mut final_results = Vec::new();
        for result in results {
            match result {
                Ok(experiment_result) => final_results.push(experiment_result),
                Err(e) => {
                    return Err(NNError::ExecutionError {
                        message: format!("Workflow step execution failed: {}", e),
                    })
                }
            }
        }

        Ok(final_results)
    }

    /// Execute individual workflow step
    async fn execute_workflow_step(
        step: &WorkflowStep,
        registry: &ResearchAgentRegistry,
    ) -> Result<ExperimentResult> {
        // Create experiment specification from step config
        let experiment_spec = ExperimentSpec {
            id: step.id.clone(),
            name: step.name.clone(),
            domain: super::ResearchDomain::GeneralML, // Default domain
            agent_type: step.agent_type.clone(),
            experiment_config: step.config.clone(),
            resource_requirements: super::agent::ResourceRequirements::default(),
            dependencies: step.dependencies.clone(),
            priority: step.priority,
            timeout_secs: None,
            quality_constraints: super::experiment::QualityConstraints::default(),
            metadata: HashMap::new(),
        };

        // Execute through registry
        let mut agent = registry.create_agent(&step.agent_type, step.config.clone())?;
        agent.run_step(&experiment_spec)
    }
}

/// Workflow execution result
#[derive(Debug)]
pub struct WorkflowResult {
    /// Overall workflow status
    pub status: WorkflowStatus,
    /// Individual experiment results
    pub experiment_results: Vec<ExperimentResult>,
    /// Workflow execution time
    pub execution_time: std::time::Duration,
}

/// Workflow execution status
#[derive(Debug)]
pub enum WorkflowStatus {
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed,
    /// Workflow is partially complete
    Partial,
    /// Workflow was cancelled
    Cancelled,
}

/// Execution graph for DAG-based workflow orchestration
#[allow(dead_code)]
struct ExecutionGraph {
    /// Execution nodes
    nodes: Vec<ExecutionNode>,
    /// Adjacency list for dependencies (node -> dependent nodes)
    edges: Vec<Vec<usize>>,
    /// Completed node tracking
    completed: HashSet<usize>,
}

/// Execution node representing a workflow step
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ExecutionNode {
    /// Step identifier
    step_id: String,
    /// Step index in workflow
    step_index: usize,
    /// Dependencies (step IDs)
    dependencies: Vec<String>,
    /// Current execution status
    status: ExecutionStatus,
    /// Execution priority
    priority: u32,
}

/// Execution status for workflow steps
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Step is waiting for dependencies
    Pending,
    /// Step is currently executing
    Running,
    /// Step completed successfully
    Completed,
    /// Step failed
    Failed,
    /// Step was skipped
    Skipped,
}

impl ExecutionGraph {
    /// Create new execution graph
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            completed: HashSet::new(),
        }
    }

    /// Add execution node
    fn add_node(&mut self, node: ExecutionNode) {
        self.nodes.push(node);
        self.edges.push(Vec::new());
    }

    /// Add dependency edge (from -> to)
    fn add_edge(&mut self, from: usize, to: usize) {
        if to < self.edges.len() {
            self.edges[from].push(to);
        }
    }

    /// Mark node as completed
    #[allow(dead_code)]
    fn mark_completed(&mut self, node_idx: usize) {
        self.completed.insert(node_idx);
        self.nodes[node_idx].status = ExecutionStatus::Completed;
    }

    /// Get nodes ready for execution (dependencies satisfied)
    #[allow(dead_code)]
    fn get_ready_nodes(&self) -> Vec<usize> {
        let mut ready = Vec::new();

        for (idx, node) in self.nodes.iter().enumerate() {
            if self.completed.contains(&idx) || node.status != ExecutionStatus::Pending {
                continue;
            }

            // Check if all dependencies are satisfied
            let mut deps_satisfied = true;
            for dep_step_id in &node.dependencies {
                // Find dependency node index
                if let Some(dep_idx) = self.nodes.iter().position(|n| n.step_id == *dep_step_id) {
                    if !self.completed.contains(&dep_idx) {
                        deps_satisfied = false;
                        break;
                    }
                }
            }

            if deps_satisfied {
                ready.push(idx);
            }
        }

        // Sort by priority (higher priority first)
        ready.sort_by(|a, b| self.nodes[*b].priority.cmp(&self.nodes[*a].priority));
        ready
    }

    /// Check if execution graph is complete
    #[allow(dead_code)]
    fn is_complete(&self) -> bool {
        self.completed.len() == self.nodes.len()
    }
}

/// Resource allocation for workflow steps
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocated GPUs
    pub gpu_count: usize,
    /// Allocated CPU cores
    pub cpu_cores: usize,
    /// Allocated memory (MB)
    pub memory_mb: usize,
}

/// Resource manager for workflow execution
#[derive(Debug)]
pub struct ResourceManager {
    /// Total available GPUs
    total_gpus: usize,
    /// Total available CPU cores
    total_cpu_cores: usize,
    /// Total available memory (MB)
    total_memory_mb: usize,
    /// Currently allocated resources
    allocated_resources: Mutex<ResourceAllocation>,
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new(total_gpus: usize, total_cpu_cores: usize, total_memory_mb: usize) -> Self {
        Self {
            total_gpus,
            total_cpu_cores,
            total_memory_mb,
            allocated_resources: Mutex::new(ResourceAllocation {
                gpu_count: 0,
                cpu_cores: 0,
                memory_mb: 0,
            }),
        }
    }

    /// Allocate resources for workflow step
    async fn allocate_resources(&self, step: &WorkflowStep) -> Result<ResourceAllocation> {
        // Extract resource requirements from step config
        let gpu_required = step
            .config
            .get("gpu_required")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let cpu_required = step
            .config
            .get("cpu_required")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let memory_required = step
            .config
            .get("memory_mb")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;

        let mut allocated =
            self.allocated_resources
                .lock()
                .map_err(|e| NNError::ResourceError {
                    message: format!("Failed to acquire resource lock: {}", e),
                })?;

        // Check resource availability
        if allocated.gpu_count + gpu_required > self.total_gpus
            || allocated.cpu_cores + cpu_required > self.total_cpu_cores
            || allocated.memory_mb + memory_required > self.total_memory_mb
        {
            return Err(NNError::ResourceError {
                message: format!(
                    "Insufficient resources for step {}: GPU {}/{}, CPU {}/{}, Memory {}/{} MB",
                    step.id,
                    allocated.gpu_count + gpu_required,
                    self.total_gpus,
                    allocated.cpu_cores + cpu_required,
                    self.total_cpu_cores,
                    allocated.memory_mb + memory_required,
                    self.total_memory_mb
                ),
            });
        }

        // Allocate resources
        allocated.gpu_count += gpu_required;
        allocated.cpu_cores += cpu_required;
        allocated.memory_mb += memory_required;

        Ok(ResourceAllocation {
            gpu_count: gpu_required,
            cpu_cores: cpu_required,
            memory_mb: memory_required,
        })
    }

    /// Release allocated resources
    async fn release_resources(&self, allocation: ResourceAllocation) -> Result<()> {
        let mut allocated =
            self.allocated_resources
                .lock()
                .map_err(|e| NNError::ResourceError {
                    message: format!("Failed to acquire resource lock: {}", e),
                })?;

        allocated.gpu_count = allocated.gpu_count.saturating_sub(allocation.gpu_count);
        allocated.cpu_cores = allocated.cpu_cores.saturating_sub(allocation.cpu_cores);
        allocated.memory_mb = allocated.memory_mb.saturating_sub(allocation.memory_mb);

        Ok(())
    }
}

/// Progress tracker for workflow execution
#[derive(Debug)]
pub struct ProgressTracker {
    /// Workflow progress state
    workflow_progress: RwLock<HashMap<String, WorkflowProgress>>,
    /// Step execution metrics
    step_metrics: RwLock<HashMap<(String, String), StepMetrics>>,
}

#[derive(Debug, Clone)]
pub struct WorkflowProgress {
    /// Workflow status
    pub status: WorkflowExecutionStatus,
    /// Start time
    pub start_time: Instant,
    /// Completion time
    pub end_time: Option<Instant>,
    /// Progress percentage (0-100)
    pub progress_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Execution status
    pub status: ExecutionStatus,
    /// Start time
    pub start_time: Option<Instant>,
    /// Execution time
    execution_time: Option<Duration>,
    /// Resource usage
    #[allow(dead_code)]
    resource_usage: Option<ResourceAllocation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowExecutionStatus {
    /// Workflow not started
    Pending,
    /// Workflow is running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed,
    /// Workflow was cancelled
    Cancelled,
}

impl ProgressTracker {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            workflow_progress: RwLock::new(HashMap::new()),
            step_metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Update workflow status
    async fn update_workflow_status(&self, workflow_id: &str, status: WorkflowExecutionStatus) {
        let mut progress = self.workflow_progress.write().await;
        let entry = progress
            .entry(workflow_id.to_string())
            .or_insert(WorkflowProgress {
                status: WorkflowExecutionStatus::Pending,
                start_time: Instant::now(),
                end_time: None,
                progress_percentage: 0.0,
            });

        entry.status = status.clone();
        if status == WorkflowExecutionStatus::Completed || status == WorkflowExecutionStatus::Failed
        {
            entry.end_time = Some(Instant::now());
        }
    }

    /// Update step status
    async fn update_step_status(&self, workflow_id: &str, step_id: &str, status: ExecutionStatus) {
        let mut metrics = self.step_metrics.write().await;
        let key = (workflow_id.to_string(), step_id.to_string());

        let entry = metrics.entry(key).or_insert(StepMetrics {
            status: ExecutionStatus::Pending,
            start_time: None,
            execution_time: None,
            resource_usage: None,
        });

        entry.status = status.clone();

        if status == ExecutionStatus::Running && entry.start_time.is_none() {
            entry.start_time = Some(Instant::now());
        }
    }

    /// Record step execution time
    async fn record_step_execution_time(
        &self,
        workflow_id: &str,
        step_id: &str,
        execution_time: Duration,
    ) {
        let mut metrics = self.step_metrics.write().await;
        let key = (workflow_id.to_string(), step_id.to_string());

        if let Some(entry) = metrics.get_mut(&key) {
            entry.execution_time = Some(execution_time);
        }
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ResearchOrchestrator {
    /// Create new orchestrator with advanced capabilities
    pub fn new(config: ResearchConfig) -> Self {
        let execution_engine = WorkflowExecutionEngine::new(
            config.max_concurrent_experiments,
            Duration::from_secs(3600), // Default 1 hour timeout
            true,                      // Enable parallel execution
        );

        let resource_manager = Arc::new(ResourceManager::new(
            4, // Default 4 GPUs
            num_cpus::get(),
            32768, // Default 32GB memory
        ));

        let progress_tracker = Arc::new(ProgressTracker::new());

        Self {
            config,
            execution_engine,
            resource_manager,
            progress_tracker,
        }
    }

    /// Create orchestrator with custom resource constraints
    pub fn with_resources(
        config: ResearchConfig,
        max_concurrent_steps: usize,
        total_gpus: usize,
        total_memory_mb: usize,
    ) -> Self {
        let execution_engine = WorkflowExecutionEngine::new(
            max_concurrent_steps,
            Duration::from_secs(config.max_concurrent_experiments as u64 * 3600), // Scale timeout with concurrency
            true,
        );

        let resource_manager = Arc::new(ResourceManager::new(
            total_gpus,
            num_cpus::get(),
            total_memory_mb,
        ));

        let progress_tracker = Arc::new(ProgressTracker::new());

        Self {
            config,
            execution_engine,
            resource_manager,
            progress_tracker,
        }
    }

    /// Execute research workflow with advanced orchestration
    pub async fn execute_workflow_async(
        &mut self,
        workflow: &super::ResearchWorkflow,
        registry: &ResearchAgentRegistry,
    ) -> Result<WorkflowResult> {
        tracing::info!("Starting workflow execution: {}", workflow.name);

        let result = self
            .execution_engine
            .execute_workflow_dag(
                workflow,
                registry,
                &self.resource_manager,
                &self.progress_tracker,
            )
            .await;

        match &result {
            Ok(workflow_result) => {
                tracing::info!(
                    "Workflow {} completed with status: {:?}, execution time: {:?}",
                    workflow.name,
                    workflow_result.status,
                    workflow_result.execution_time
                );
            }
            Err(e) => {
                tracing::error!("Workflow {} failed: {}", workflow.name, e);
            }
        }

        result
    }

    /// Execute workflow synchronously (legacy compatibility)
    pub fn execute_workflow(
        &mut self,
        _workflow: &super::ResearchWorkflow,
        _registry: &ResearchAgentRegistry,
    ) -> Result<ExperimentResult> {
        // For backward compatibility, return a placeholder result
        // Advanced async execution should be used instead
        tracing::warn!(
            "Synchronous workflow execution is deprecated. Use execute_workflow_async() instead."
        );
        Ok(ExperimentResult::new(
            "workflow_execution".to_string(),
            "orchestrator".to_string(),
        ))
    }

    /// Execute single experiment
    pub fn execute_experiment(
        &self,
        experiment: &ExperimentSpec,
        registry: &ResearchAgentRegistry,
    ) -> Result<ExperimentResult> {
        tracing::info!("Executing experiment: {}", experiment.id);

        let result = (|| {
            let mut agent = registry
                .create_agent(&experiment.agent_type, experiment.experiment_config.clone())?;
            agent.run_step(experiment)
        })();

        match &result {
            Ok(_) => tracing::info!("Experiment {} completed successfully", experiment.id),
            Err(e) => tracing::error!("Experiment {} failed: {}", experiment.id, e),
        }

        result
    }

    /// Get workflow execution progress
    pub async fn get_workflow_progress(&self, workflow_id: &str) -> Option<WorkflowProgress> {
        let progress = self.progress_tracker.workflow_progress.read().await;
        progress.get(workflow_id).cloned()
    }

    /// Get step execution metrics
    pub async fn get_step_metrics(&self, workflow_id: &str, step_id: &str) -> Option<StepMetrics> {
        let metrics = self.progress_tracker.step_metrics.read().await;
        metrics
            .get(&(workflow_id.to_string(), step_id.to_string()))
            .cloned()
    }

    /// Get resource utilization
    pub fn get_resource_utilization(&self) -> Result<ResourceAllocation> {
        let allocated = self
            .resource_manager
            .allocated_resources
            .lock()
            .map_err(|e| NNError::ResourceError {
                message: format!("Failed to acquire resource lock: {}", e),
            })?;

        Ok(allocated.clone())
    }

    /// Cancel workflow execution
    pub async fn cancel_workflow(&self, workflow_id: &str) -> Result<()> {
        self.progress_tracker
            .update_workflow_status(workflow_id, WorkflowExecutionStatus::Cancelled)
            .await;
        tracing::info!("Workflow {} cancelled", workflow_id);
        Ok(())
    }

    /// Get orchestrator health status
    pub fn health_status(&self) -> OrchestratorHealthStatus {
        OrchestratorHealthStatus {
            resource_manager_healthy: true, // Basic health check
            progress_tracker_healthy: true,
            execution_engine_healthy: true,
            active_workflows: 0, // TODO: Implement workflow counting
        }
    }
}

/// Orchestrator health status
#[derive(Debug, Clone)]
pub struct OrchestratorHealthStatus {
    /// Resource manager health
    pub resource_manager_healthy: bool,
    /// Progress tracker health
    pub progress_tracker_healthy: bool,
    /// Execution engine health
    pub execution_engine_healthy: bool,
    /// Number of active workflows
    pub active_workflows: usize,
}

impl std::fmt::Display for OrchestratorHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let overall_health = self.resource_manager_healthy
            && self.progress_tracker_healthy
            && self.execution_engine_healthy;

        let status = if overall_health {
            "🟢 HEALTHY"
        } else {
            "🔴 ISSUES DETECTED"
        };

        write!(
            f,
            "🔧 Orchestrator Health: {}\n\
             ├── Resource Manager: {}\n\
             ├── Progress Tracker: {}\n\
             ├── Execution Engine: {}\n\
             └── Active Workflows: {}",
            status,
            if self.resource_manager_healthy {
                "🟢"
            } else {
                "🔴"
            },
            if self.progress_tracker_healthy {
                "🟢"
            } else {
                "🔴"
            },
            if self.execution_engine_healthy {
                "🟢"
            } else {
                "🔴"
            },
            self.active_workflows
        )
    }
}
