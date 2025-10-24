# Unified Research Framework Guide

## Overview

The Unified Research Framework consolidates research agents across Neural Architecture Search (NAS), Hyperparameter Optimization (HPO), and Meta-Learning into a cohesive system with shared abstractions, experiment orchestration, and knowledge transfer capabilities.

## Key Components

### 1. Research Agent Traits

All research agents implement the `ResearchAgent` trait:

```rust
pub trait ResearchAgent: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn agent_type(&self) -> AgentType;
    fn metadata(&self) -> AgentMetadata;
    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult>;
    // ... additional methods
}
```

### 2. Unified Experiment Specifications

Experiments are defined using `ExperimentSpec`:

```rust
let experiment = ExperimentSpec::new(
    "exp_001".to_string(),
    "Architecture Search".to_string(),
    ResearchDomain::ComputerVision,
    "nas_agent".to_string(),
)
.with_config(json!({"budget": 100}))
.with_resources(ResourceRequirements {
    gpu_memory_gb: 8.0,
    cpu_cores: 4,
    ..Default::default()
});
```

### 3. Agent Registry

The registry manages agent factories:

```rust
let mut registry = ResearchAgentRegistry::new();
registry.register::<HPOAgentAdapter<HPOAgent>>("hpo_agent")?;
```

### 4. Workflow Templates

Predefined workflows for common research patterns:

```rust
// NAS-HPO collaboration
let workflow = WorkflowTemplate::nas_hpo_collaboration("accuracy");

// Comprehensive AutoML
let automl = WorkflowTemplate::comprehensive_automl();

// Comparative benchmarking
let benchmark = WorkflowTemplate::comparative_benchmark();
```

## Integrating Existing Agents

### HPO Agent Integration

```rust
use coeus_nn::hpo::BayesianOptimizer;
use coeus_nn::research::{ResearchAgent, ExperimentSpec, ExperimentResult};

struct HPOAgentAdapter {
    optimizer: BayesianOptimizer,
    agent_id: String,
}

impl ResearchAgent for HPOAgentAdapter {
    fn id(&self) -> &str { &self.agent_id }

    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        // Convert experiment config to HPO config
        let hpo_config: HyperparameterConfig = serde_json::from_value(
            experiment.experiment_config.clone()
        )?;

        // Run HPO optimization
        let (config, value) = self.optimizer.suggest_and_observe(hpo_config)?;

        // Return unified result
        let mut result = ExperimentResult::new(
            experiment.id.clone(),
            self.agent_id.clone()
        );
        result.mark_completed(value);
        Ok(result)
    }

    // ... implement other trait methods
}
```

### NAS Agent Integration

```rust
use coeus_nn::nas::DartsNAS;
use coeus_nn::research::ResearchAgent;

struct NASAgentAdapter {
    nas_agent: DartsNAS,
    agent_id: String,
}

impl ResearchAgent for NASAgentAdapter {
    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        // Extract architecture space from experiment config
        let search_space: ArchitectureSpace = serde_json::from_value(
            experiment.experiment_config["search_space"].clone()
        )?;

        // Run NAS iteration
        let fitness_fn = |architecture: &Architecture| -> f64 {
            // Evaluate architecture (simplified)
            evaluate_architecture(architecture)
        };

        let performance = self.nas_agent.search_step(fitness_fn)?;

        let mut result = ExperimentResult::new(
            experiment.id.clone(),
            self.agent_id.clone()
        );
        result.mark_completed(performance);
        Ok(result)
    }

    // ... implement other trait methods
}
```

### Meta-Learning Agent Integration

```rust
use coeus_nn::meta::MAML;
use coeus_nn::research::ResearchAgent;

struct MetaAgentAdapter<M: Module<B, S, T>> {
    maml_agent: MAML<M>,
    agent_id: String,
}

impl<M: Module<B, S, T>> ResearchAgent for MetaAgentAdapter<M> {
    fn run_step(&mut self, experiment: &ExperimentSpec) -> Result<ExperimentResult> {
        // Extract task distribution from config
        let tasks: Vec<Task<B, S, T>> = parse_tasks_from_config(
            &experiment.experiment_config
        )?;

        // Run meta-learning step
        let meta_loss = self.maml_agent.meta_step(&tasks)?;

        let mut result = ExperimentResult::new(
            experiment.id.clone(),
            self.agent_id.clone()
        );
        result.mark_completed(1.0 - meta_loss); // Convert loss to performance
        Ok(result)
    }

    // ... implement other trait methods
}
```

## Knowledge Transfer

### Insight Generation

Agents can generate insights for knowledge transfer:

```rust
impl ResearchAgent for MyAgent {
    fn generate_insights(&self) -> Vec<ResearchInsight> {
        vec![ResearchInsight {
            id: "insight_001".to_string(),
            agent_type: "my_agent".to_string(),
            domains: vec![ResearchDomain::GeneralML],
            performance_impact: 0.15,
            confidence: 0.85,
            knowledge_data: json!({
                "optimal_param": "value",
                "confidence_interval": [0.1, 0.2]
            }),
            timestamp: Instant::now(),
        }]
    }
}
```

### Accessing Cross-Agent Insights

```rust
// Get insights relevant to a domain
let ml_insights = framework.get_domain_insights(&ResearchDomain::GeneralML);

// Add new insight
framework.add_insight(my_insight);

// Insights are automatically shared with compatible agents
```

## Workflow Orchestration

### Running Workflows

```rust
let workflow = WorkflowTemplate::nas_hpo_collaboration("accuracy");
let result = framework.execute_workflow(&workflow)?;
println!("Final performance: {:.3}", result.final_performance);
```

### Custom Workflows

```rust
let custom_workflow = ResearchWorkflow {
    id: "my_workflow".to_string(),
    name: "Custom Research Pipeline".to_string(),
    domain: ResearchDomain::AutoML,
    steps: vec![
        WorkflowStep {
            id: "preprocessing".to_string(),
            agent_type: "data_agent".to_string(),
            config: json!({"normalize": true}),
            dependencies: vec![],
            priority: 10,
        },
        WorkflowStep {
            id: "optimization".to_string(),
            agent_type: "hpo_agent".to_string(),
            config: json!({"method": "bayesian"}),
            dependencies: vec!["preprocessing".to_string()],
            priority: 9,
        },
    ],
    ..Default::default()
};
```

## Resource Management

### Resource Requirements

Each agent specifies its resource needs:

```rust
impl ResearchAgent for MyAgent {
    fn get_resource_requirements(&self) -> ResourceRequirements {
        ResourceRequirements {
            cpu_cores: 4,
            gpu_memory_gb: 8.0,
            system_memory_gb: 16.0,
            storage_gb: 50.0,
            estimated_time_secs: 1800, // 30 minutes
        }
    }
}
```

### Quality Constraints

Experiments can define quality requirements:

```rust
let experiment = ExperimentSpec::new(...)
    .with_quality_constraints(QualityConstraints {
        min_performance: Some(0.85),
        max_variance: Some(0.05),
        significance_level: Some(0.05),
        ..Default::default()
    });
```

## Monitoring and Reporting

### Framework Metrics

```rust
let metrics = framework.get_metrics();
println!("Total experiments: {}", metrics.total_experiments);
println!("Success rate: {:.1}%", metrics.successful_experiments as f64 /
                                  metrics.total_experiments as f64 * 100.0);

let report = framework.generate_report();
println!("{}", report); // Formatted summary
```

### Experiment Results

```rust
let result = agent.run_step(&experiment)?;

// Check quality constraints
if result.meets_quality_constraints(&experiment.quality_constraints) {
    println!("Experiment meets quality standards");
}

// Get detailed statistics
println!("Duration: {:.2}s", result.duration().as_secs_f64());
println!("Performance: {:.4}", result.final_performance);

// Access insights
for insight in &result.insights {
    println!("Generated insight: {}", insight.id);
}
```

## Best Practices

### Agent Design

1. **Immutability**: Prefer functional updates over mutable state
2. **Error Handling**: Use structured error types with context
3. **Resource Awareness**: Declare resource requirements accurately
4. **Insight Generation**: Generate actionable insights when possible
5. **Domain Specification**: Clearly define supported research domains

### Framework Usage

1. **Workflow Composition**: Start with predefined templates, customize as needed
2. **Resource Planning**: Consider total resource requirements of workflows
3. **Monitoring**: Regularly check framework metrics and reports
4. **Knowledge Transfer**: Design agents to both consume and produce insights

### Integration Patterns

1. **Adapters**: Use adapter pattern to integrate existing agents
2. **Factories**: Implement factory pattern for agent creation
3. **Configuration**: Use JSON configuration for agent parameters
4. **Validation**: Validate experiment specifications before execution

## API Reference

### Core Types

- `ResearchAgent`: Core agent trait
- `ExperimentSpec`: Experiment definition
- `ExperimentResult`: Execution results
- `ResearchWorkflow`: Workflow definition
- `UnifiedResearchFramework`: Main framework interface

### Agent Types

- `AgentType::NAS`: Neural Architecture Search
- `AgentType::HPO`: Hyperparameter Optimization
- `AgentType::MetaLearning`: Meta-Learning
- `AgentType::Hybrid`: Multi-agent coordination

### Research Domains

- `ComputerVision`: Vision tasks
- `NLP`: Natural language processing
- `ReinforcementLearning`: RL tasks
- `GeneralML`: General machine learning
- `MetaLearning`: Meta-learning tasks
- `AutoML`: Automated ML pipelines

This unified framework provides a foundation for coordinated, efficient research across multiple domains while maintaining the specialized capabilities of individual research agents.
