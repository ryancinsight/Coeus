//! Advanced Research Workflow Automation Example
//!
//! This example demonstrates the comprehensive research workflow automation
//! capabilities of the Coeus framework, including:
//!
//! - Configuration-driven workflow specifications
//! - DAG-based parallel execution with resource management
//! - Real-time progress monitoring and failure recovery
//! - Declarative workflow definition via YAML/JSON

use std::path::Path;
use tokio;
use nn::research::{
    UnifiedResearchFramework, WorkflowSpec, WorkflowMetadata, StepSpec,
    ResearchDomain, ResourceRequirements, RetryConfig, WorkflowConfig,
    ExecutionMode, FailureStrategy, WorkflowLoader
};
use nn::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Advanced Research Workflow Automation Demo");
    println!("==============================================\n");

    // Initialize the unified research framework
    let mut framework = UnifiedResearchFramework::new();
    println!("✅ Unified Research Framework initialized");

    // Example 1: Template-based workflow execution
    println!("\n📋 Example 1: Template-based NAS-HPO Workflow");
    println!("------------------------------------------------");
    await example_template_workflow(&mut framework).await?;

    // Example 2: Configuration-driven workflow from YAML
    println!("\n📄 Example 2: YAML Configuration-Driven Workflow");
    println!("-------------------------------------------------");
    await example_yaml_workflow(&mut framework).await?;

    // Example 3: Advanced orchestration with monitoring
    println!("\n📊 Example 3: Advanced Orchestration with Monitoring");
    println!("-----------------------------------------------------");
    await example_advanced_orchestration(&mut framework).await?;

    // Example 4: Resource management and failure recovery
    println!("\n🛡️  Example 4: Resource Management & Failure Recovery");
    println!("-----------------------------------------------------");
    await example_resource_management(&mut framework).await?;

    println!("\n🎉 All workflow automation examples completed successfully!");
    println!("\nKey Features Demonstrated:");
    println!("• DAG-based parallel execution with dependency management");
    println!("• Resource-aware scheduling with GPU/CPU/memory constraints");
    println!("• Real-time progress monitoring and metrics collection");
    println!("• Declarative workflow specification via YAML/JSON");
    println!("• Comprehensive failure recovery and retry mechanisms");
    println!("• Template-based workflow composition");

    Ok(())
}

/// Demonstrate template-based workflow execution
async fn example_template_workflow(framework: &mut UnifiedResearchFramework) -> Result<()> {
    println!("Using predefined NAS-HPO collaboration template...");

    // Create a template-based workflow
    let workflow = nn::research::WorkflowTemplate::nas_hpo_collaboration("accuracy");

    println!("Workflow '{}' created with {} steps", workflow.name, workflow.steps.len());

    // Execute the workflow
    let result = framework.execute_workflow_async(&workflow).await?;

    println!("✅ Workflow completed in {:?}", result.execution_time);
    println!("   Status: {:?}", result.status);
    println!("   Experiments completed: {}", result.experiment_results.len());

    Ok(())
}

/// Demonstrate YAML configuration-driven workflow
async fn example_yaml_workflow(framework: &mut UnifiedResearchFramework) -> Result<()> {
    println!("Creating workflow from YAML specification...");

    // Create a YAML workflow specification
    let yaml_content = r#"
metadata:
  id: "custom_ml_pipeline"
  name: "Custom ML Pipeline"
  description: "A custom machine learning pipeline with data processing and model training"
  domain: "AutoML"
  version: "1.0.0"
  author: "automation_example"
  tags: ["example", "custom"]

steps:
  - id: "data_preprocessing"
    name: "Data Preprocessing"
    agent_type: "data_processor"
    depends_on: []
    priority: 10
    resources:
      cpu_required: 2
      memory_mb: 2048
    config:
      task: "normalize"
      input_format: "csv"

  - id: "feature_engineering"
    name: "Feature Engineering"
    agent_type: "feature_engineer"
    depends_on: ["data_preprocessing"]
    priority: 9
    resources:
      cpu_required: 4
      memory_mb: 4096
    config:
      method: "auto"
      max_features: 100

  - id: "model_training"
    name: "Model Training"
    agent_type: "trainer"
    depends_on: ["feature_engineering"]
    priority: 8
    resources:
      gpu_required: 1
      cpu_required: 8
      memory_mb: 16384
    retry:
      max_attempts: 3
      delay_seconds: 60
    config:
      algorithm: "xgboost"
      objective: "binary_classification"

  - id: "model_evaluation"
    name: "Model Evaluation"
    agent_type: "evaluator"
    depends_on: ["model_training"]
    priority: 7
    resources:
      cpu_required: 2
      memory_mb: 2048
    config:
      metrics: ["accuracy", "precision", "recall", "f1_score"]

config:
  constraints:
    max_execution_time: 3600
    resource_limits:
      total_gpus: 2
      total_memory_mb: 32768
  parameters:
    dataset_path: "/data/ml_dataset.csv"
    output_dir: "/results"
  execution_mode: "Parallel"
  failure_strategy: "FailFast"
"#;

    // Write to temporary file
    let temp_path = "temp_workflow.yaml";
    std::fs::write(temp_path, yaml_content)?;

    // Load and execute workflow from YAML
    let result = framework.execute_workflow_from_yaml(temp_path).await?;

    println!("✅ YAML workflow completed in {:?}", result.execution_time);
    println!("   Steps executed: {}", result.experiment_results.len());

    // Clean up
    std::fs::remove_file(temp_path)?;

    Ok(())
}

/// Demonstrate advanced orchestration with monitoring
async fn example_advanced_orchestration(framework: &mut UnifiedResearchFramework) -> Result<()> {
    println!("Demonstrating advanced orchestration capabilities...");

    // Create a complex workflow with monitoring
    let workflow_spec = WorkflowSpec {
        metadata: WorkflowMetadata {
            id: "monitored_pipeline".to_string(),
            name: "Monitored Research Pipeline".to_string(),
            description: "A pipeline with comprehensive monitoring and metrics".to_string(),
            domain: ResearchDomain::AutoML,
            version: "1.0.0".to_string(),
            author: "monitoring_example".to_string(),
            tags: vec!["monitoring".to_string(), "advanced".to_string()],
        },
        steps: vec![
            StepSpec {
                id: "data_ingestion".to_string(),
                name: "Data Ingestion".to_string(),
                agent_type: "data_ingestor".to_string(),
                config: serde_json::json!({"source": "s3", "format": "parquet"}),
                depends_on: vec![],
                priority: 10,
                resources: ResourceRequirements {
                    gpu_required: 0,
                    cpu_required: 2,
                    memory_mb: 1024,
                    max_execution_time: Some(300),
                },
                retry: RetryConfig::default(),
                condition: None,
            },
            StepSpec {
                id: "preprocessing".to_string(),
                name: "Preprocessing".to_string(),
                agent_type: "preprocessor".to_string(),
                config: serde_json::json!({"method": "standard", "impute_missing": true}),
                depends_on: vec!["data_ingestion".to_string()],
                priority: 9,
                resources: ResourceRequirements {
                    gpu_required: 0,
                    cpu_required: 4,
                    memory_mb: 4096,
                    max_execution_time: Some(600),
                },
                retry: RetryConfig {
                    max_attempts: 2,
                    delay_seconds: 30,
                    backoff_multiplier: 1.5,
                },
                condition: None,
            },
            StepSpec {
                id: "training".to_string(),
                name: "Model Training".to_string(),
                agent_type: "trainer".to_string(),
                config: serde_json::json!({"model": "random_forest", "hyperparams": {"n_estimators": 100}}),
                depends_on: vec!["preprocessing".to_string()],
                priority: 8,
                resources: ResourceRequirements {
                    gpu_required: 1,
                    cpu_required: 8,
                    memory_mb: 16384,
                    max_execution_time: Some(1800),
                },
                retry: RetryConfig {
                    max_attempts: 3,
                    delay_seconds: 120,
                    backoff_multiplier: 2.0,
                },
                condition: None,
            },
        ],
        config: WorkflowConfig {
            constraints: serde_json::from_value(serde_json::json!({
                "max_execution_time": 3600,
                "resource_limits": {
                    "total_gpus": 2,
                    "total_memory_mb": 32768
                }
            })).unwrap(),
            parameters: serde_json::json!({
                "input_data": "/data/input",
                "output_models": "/models",
                "logs_dir": "/logs"
            }),
            execution_mode: ExecutionMode::Parallel,
            failure_strategy: FailureStrategy::FailFast,
        },
        extends: None,
    };

    let workflow = WorkflowLoader::spec_to_workflow(workflow_spec)?;

    // Start workflow execution
    println!("Starting monitored workflow execution...");
    let workflow_id = workflow.id.clone();

    // Execute workflow in background task for monitoring demo
    let monitor_handle = tokio::spawn(async move {
        let result = framework.execute_workflow_async(&workflow).await;

        match result {
            Ok(workflow_result) => {
                println!("✅ Workflow completed successfully");
                println!("   Execution time: {:?}", workflow_result.execution_time);
                println!("   Status: {:?}", workflow_result.status);
            }
            Err(e) => {
                println!("❌ Workflow failed: {}", e);
            }
        }
    });

    // Monitor progress in real-time
    println!("Monitoring workflow progress...");
    for i in 0..10 {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        if let Some(progress) = framework.get_workflow_progress(&workflow_id).await {
            println!("Progress update {}: Status={:?}, Progress={:.1}%",
                    i + 1, progress.status, progress.progress_percentage);

            // Check step metrics
            for step in &workflow.steps {
                if let Some(metrics) = framework.get_step_metrics(&workflow_id, &step.id).await {
                    if matches!(metrics.status, nn::research::ExecutionStatus::Completed) {
                        println!("   Step '{}' completed in {:?}", step.name, metrics.execution_time);
                    }
                }
            }
        }
    }

    // Wait for completion
    monitor_handle.await?;

    Ok(())
}

/// Demonstrate resource management and failure recovery
async fn example_resource_management(framework: &mut UnifiedResearchFramework) -> Result<()> {
    println!("Demonstrating resource management and failure recovery...");

    // Create workflow with resource constraints
    let workflow_spec = WorkflowSpec {
        metadata: WorkflowMetadata {
            id: "resource_managed".to_string(),
            name: "Resource Managed Workflow".to_string(),
            description: "Workflow with strict resource management".to_string(),
            domain: ResearchDomain::AutoML,
            version: "1.0.0".to_string(),
            author: "resource_example".to_string(),
            tags: vec!["resources".to_string(), "failure_recovery".to_string()],
        },
        steps: vec![
            StepSpec {
                id: "gpu_intensive_task".to_string(),
                name: "GPU Intensive Task".to_string(),
                agent_type: "gpu_worker".to_string(),
                config: serde_json::json!({"task": "gpu_computation", "intensity": "high"}),
                depends_on: vec![],
                priority: 10,
                resources: ResourceRequirements {
                    gpu_required: 1,
                    cpu_required: 4,
                    memory_mb: 8192,
                    max_execution_time: Some(300),
                },
                retry: RetryConfig {
                    max_attempts: 2,
                    delay_seconds: 30,
                    backoff_multiplier: 2.0,
                },
                condition: None,
            },
            StepSpec {
                id: "cpu_parallel_task".to_string(),
                name: "CPU Parallel Task".to_string(),
                agent_type: "cpu_worker".to_string(),
                config: serde_json::json!({"task": "cpu_computation", "parallelism": 8}),
                depends_on: vec![],
                priority: 9,
                resources: ResourceRequirements {
                    gpu_required: 0,
                    cpu_required: 8,
                    memory_mb: 4096,
                    max_execution_time: Some(600),
                },
                retry: RetryConfig::default(),
                condition: None,
            },
            StepSpec {
                id: "memory_intensive_task".to_string(),
                name: "Memory Intensive Task".to_string(),
                agent_type: "memory_worker".to_string(),
                config: serde_json::json!({"task": "memory_processing", "data_size": "large"}),
                depends_on: vec!["cpu_parallel_task".to_string()],
                priority: 8,
                resources: ResourceRequirements {
                    gpu_required: 0,
                    cpu_required: 2,
                    memory_mb: 16384,
                    max_execution_time: Some(900),
                },
                retry: RetryConfig {
                    max_attempts: 3,
                    delay_seconds: 60,
                    backoff_multiplier: 1.5,
                },
                condition: None,
            },
        ],
        config: WorkflowConfig {
            constraints: serde_json::from_value(serde_json::json!({
                "max_execution_time": 1800,
                "resource_limits": {
                    "total_gpus": 1,  // Limited GPU resources
                    "total_memory_mb": 24576  // Limited memory
                }
            })).unwrap(),
            parameters: serde_json::json!({}),
            execution_mode: ExecutionMode::Parallel,
            failure_strategy: FailureStrategy::RetryFailed,
        },
        extends: None,
    };

    let workflow = WorkflowLoader::spec_to_workflow(workflow_spec)?;

    println!("Resource constraints:");
    println!("  Max GPUs: 1");
    println!("  Max Memory: 24576 MB");
    println!("  Execution mode: Parallel with failure retry");

    // Check orchestrator health before execution
    let health = framework.get_orchestrator_health();
    println!("Orchestrator health: {}", health);

    // Execute workflow
    let result = framework.execute_workflow_async(&workflow).await?;

    println!("✅ Resource-managed workflow completed");
    println!("   Status: {:?}", result.status);
    println!("   Execution time: {:?}", result.execution_time);
    println!("   Experiments: {}", result.experiment_results.len());

    // Check final resource utilization
    let utilization = framework.orchestrator.get_resource_utilization()?;
    println!("Final resource utilization: GPU={}, CPU={}, Memory={}MB",
            utilization.gpu_count, utilization.cpu_cores, utilization.memory_mb);

    Ok(())
}










