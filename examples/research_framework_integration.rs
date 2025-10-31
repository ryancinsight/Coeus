//! Unified Research Framework Integration Example
//!
//! This example demonstrates how to use the unified research framework
//! to consolidate and coordinate research agents across NAS, HPO, and meta-learning.
//!
//! This example shows:
//! 1. Setting up the unified framework
//! 2. Registering existing research agents with adapters
//! 3. Running coordinated experiments
//! 4. Knowledge transfer between agents
//! 5. Workflow orchestration

use std::collections::HashMap;
use nn::research::{
    UnifiedResearchFramework, ResearchConfig, ResearchDomain, ResearchWorkflow, WorkflowTemplate,
    ResearchAgent, ResearchAgentFactory, AgentType, ExperimentSpec, ExperimentResult,
};
use serde_json::json;

/// Example research agent adapter for existing HPO system
struct HPOAgentAdapter<H> {
    inner_agent: H,
    agent_id: String,
    domain: ResearchDomain,
}

impl<H: nn::research::ResearchAgent> ResearchAgent for HPOAgentAdapter<H> {
    fn id(&self) -> &str { &self.agent_id }
    fn name(&self) -> &str { "HPO Agent Adapter" }
    fn agent_type(&self) -> AgentType { AgentType::HPO }

    fn metadata(&self) -> super::AgentMetadata {
        super::AgentMetadata {
            version: "1.0.0".to_string(),
            supported_domains: vec![self.domain.clone()],
            resource_profile: Default::default(),
            performance_characteristics: Default::default(),
            capabilities: vec!["hpo".to_string(), "optimization".to_string()],
        }
    }

    fn supports_domain(&self, domain: &ResearchDomain) -> bool {
        domain == &self.domain
    }

    fn initialize(&mut self, _config: serde_json::Value) -> Result<(), coeus_error::NNError> {
        Ok(())
    }

    fn run_step(&mut self, _experiment: &ExperimentSpec) -> Result<ExperimentResult, coeus_error::NNError> {
        // Simulate running HPO
        let mut result = ExperimentResult::new(_experiment.id.clone(), self.id().to_string());
        result.mark_started();
        std::thread::sleep(std::time::Duration::from_millis(100));
        result.mark_completed(0.85 + rand::random::<f64>() * 0.1);
        Ok(result)
    }

    fn get_available_actions(&self) -> Vec<ExperimentSpec> {
        vec![ExperimentSpec::new(
            format!("hpo_exp_{}", self.id()),
            "HPO Optimization".to_string(),
            self.domain.clone(),
            self.id().to_string(),
        )]
    }

    fn update_with_results(&mut self, _results: &[ExperimentResult]) -> Result<(), coeus_error::NNError> {
        Ok(())
    }

    fn get_best_result(&self) -> Option<ExperimentResult> { None }
    fn get_state(&self) -> Result<serde_json::Value, coeus_error::NNError> { Ok(json!({"state": "active"})) }
    fn set_state(&mut self, _state: serde_json::Value) -> Result<(), coeus_error::NNError> { Ok(()) }
    fn is_ready(&self) -> bool { true }
    fn get_resource_requirements(&self) -> super::ResourceRequirements { Default::default() }

    fn generate_insights(&self) -> Vec<super::ResearchInsight> {
        vec![super::ResearchInsight {
            id: format!("hpo_insight_{}", self.id()),
            agent_type: "hpo".to_string(),
            domains: vec![ResearchDomain::GeneralML],
            performance_impact: 0.1,
            confidence: 0.8,
            knowledge_data: json!({"optimal_lr": 0.001, "momentum": 0.9}),
            timestamp: std::time::Instant::now(),
        }]
    }
}

/// NAS Agent Adapter
struct NASAgentAdapter<N> {
    inner_agent: N,
    agent_id: String,
}

impl<N> ResearchAgent for NASAgentAdapter<N> {
    fn id(&self) -> &str { &self.agent_id }
    fn name(&self) -> &str { "NAS Agent Adapter" }
    fn agent_type(&self) -> AgentType { AgentType::NAS }

    fn metadata(&self) -> super::AgentMetadata {
        super::AgentMetadata {
            version: "1.0.0".to_string(),
            supported_domains: vec![ResearchDomain::ComputerVision],
            resource_profile: Default::default(),
            performance_characteristics: Default::default(),
            capabilities: vec!["nas".to_string(), "architecture_search".to_string()],
        }
    }

    fn supports_domain(&self, domain: &ResearchDomain) -> bool {
        matches!(domain, ResearchDomain::ComputerVision | ResearchDomain::GeneralML)
    }

    fn initialize(&mut self, _config: serde_json::Value) -> Result<(), coeus_error::NNError> { Ok(()) }

    fn run_step(&mut self, _experiment: &ExperimentSpec) -> Result<ExperimentResult, coeus_error::NNError> {
        let mut result = ExperimentResult::new(_experiment.id.clone(), self.id().to_string());
        result.mark_started();
        std::thread::sleep(std::time::Duration::from_millis(200));
        result.mark_completed(0.75 + rand::random::<f64>() * 十五0.15);
        Ok(result)
    }

    fn get_available_actions(&self) -> Vec<ExperimentSpec> {
        vec![ExperimentSpec::new(
            format!("nas_exp_{}", self.id()),
            "NAS Architecture Search".to_string(),
            ResearchDomain::ComputerVision,
            self.id().to_string(),
        )]
    }

    fn update_with_results(&mut self, _results: &[ExperimentResult]) -> Result<(), coeus_error::NNError> { Ok(()) }
    fn get_best_result(&self) -> Option<ExperimentResult> { None }
    fn get_state(&self) -> Result<serde_json::Value, coeus_error::NNError> { Ok(json!({"state": "searching"})) }
    fn set_state(&mut self, _state: serde_json::Value) -> Result<(), coeus_error::NNError> { Ok(()) }
    fn is_ready(&self) -> bool { true }
    fn get_resource_requirements(&self) -> super::ResourceRequirements { Default::default() }

    fn generate_insights(&self) -> Vec<super::ResearchInsight> {
        vec![super::ResearchInsight {
            id: format!("nas_insight_{}", self.id()),
            agent_type: "nas".to_string(),
            domains: vec![ResearchDomain::ComputerVision],
            performance_impact: 0.15,
            confidence: 0.85,
            knowledge_data: json!({"optimal_layers": 5, "skip_connections": true}),
            timestamp: std::time::Instant::now(),
        }]
    }
}

/// Meta-Learning Agent Adapter
struct MetaAgentAdapter<M> {
    inner_agent: M,
    agent_id: String,
}

impl<M> ResearchAgent for MetaAgentAdapter<M> {
    fn id(&self) -> &str { &self.agent_id }
    fn name(&self) -> &str { "Meta-Learning Agent Adapter" }
    fn agent_type(&self) -> AgentType { AgentType::MetaLearning }

    fn metadata(&self) -> super::AgentMetadata {
        super::AgentMetadata {
            version: "1.0.0".to_string(),
            supported_domains: vec![ResearchDomain::MetaLearning],
            resource_profile: Default::default(),
            performance_characteristics: Default::default(),
            capabilities: vec!["meta_learning".to_string(), "few_shot".to_string()],
        }
    }

    fn supports_domain(&self, domain: &ResearchDomain) -> bool {
        matches!(domain, ResearchDomain::MetaLearning | ResearchDomain::GeneralML)
    }

    fn initialize(&mut self, _config: serde_json::Value) -> Result<(), coeus_error::NNError> { Ok(()) }

    fn run_step(&mut self, _experiment: &ExperimentSpec) -> Result<ExperimentResult, coeus_error::NNError> {
        let mut result = ExperimentResult::new(_experiment.id.clone(), self.id().to_string());
        result.mark_started();
        std::thread::sleep(std::time::Duration::from_millis(150));
        result.mark_completed(0.90 + rand::random::<f64>() * 0.08);
        Ok(result)
    }

    fn get_available_actions(&self) -> Vec<ExperimentSpec> {
        vec![ExperimentSpec::new(
            format!("meta_exp_{}", self.id()),
            "Meta-Learning Adaptation".to_string(),
            ResearchDomain::MetaLearning,
            self.id().to_string(),
        )]
    }

    fn update_with_results(&mut self, _results: &[ExperimentResult]) -> Result<(), coeus_error::NNError> { Ok(()) }
    fn get_best_result(&self) -> Option<ExperimentResult> { None }
    fn get_state(&self) -> Result<serde_json::Value, coeus_error::NNError> { Ok(json!({"state": "adapting"})) }
    fn set_state(&mut self, _state: serde_json::Value) -> Result<(), coeus_error::NNError> { Ok(()) }
    fn is_ready(&self) -> bool { true }
    fn get_resource_requirements(&self) -> super::ResourceRequirements { Default::default() }

    fn generate_insights(&self) -> Vec<super::ResearchInsight> {
        vec![super::ResearchInsight {
            id: format!("meta_insight_{}", self.id()),
            agent_type: "meta".to_string(),
            domains: vec![ResearchDomain::MetaLearning, ResearchDomain::GeneralML],
            performance_impact: 0.12,
            confidence: 0.9,
            knowledge_data: json!({"inner_lr": 0.01, "outer_lr": 0.001, "adaptation_steps": 5}),
            timestamp: std::time::Instant::now(),
        }]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Unified Research Framework Integration Example");
    println!("==============================================");

    // Create unified research framework
    let config = ResearchConfig::default();
    let mut framework = UnifiedResearchFramework::new(config);

    println!("\n📋 Setting up research agents...");

    // Simulate registering different types of agents
    // In practice, these would wrap existing HPO, NAS, and Meta agents
    framework.registry.register("mock_hpo_agent", json!({
        "type": "hpo",
        "domain": "GeneralML"
    }))?;

    framework.registry.register("mock_nas_agent", json!({
        "type": "nas",
        "domain": "ComputerVision"
    }))?;

    framework.registry.register("mock_meta_agent", json!({
        "type": "meta",
        "domain": "MetaLearning"
    }))?;

    println!("✅ Registered 3 research agents");

    println!("\n🔍 Available agents:");
    for agent_name in framework.registry.list_agents() {
        println!("  - {}", agent_name);
    }

    println!("\n🚀 Running research workflows...");

    // Example 1: NAS-HPO collaborative optimization
    println!("\n📊 Example 1: NAS-HPO Collaborative Optimization");
    let nas_hpo_workflow = WorkflowTemplate::nas_hpo_collaboration("accuracy");
    let result1 = framework.execute_workflow(&nas_hpo_workflow)?;
    println!("Result: {:.3} performance, {} insights generated",
             result1.final_performance, result1.insights.len());

    // Example 2: Comprehensive AutoML pipeline
    println!("\n🤖 Example 2: Comprehensive AutoML Pipeline");
    let automl_workflow = WorkflowTemplate::comprehensive_automl();
    let result2 = framework.execute_workflow(&automl_workflow)?;
    println!("Result: {:.3} performance, {} insights generated",
             result2.final_performance, result2.insights.len());

    // Example 3: Comparative benchmark
    println!("\n⚖️  Example 3: Comparative Algorithm Benchmark");
    let benchmark_workflow = WorkflowTemplate::comparative_benchmark();
    let result3 = framework.execute_workflow(&benchmark_workflow)?;
    println!("Result: {:.3} performance, {} insights generated",
             result3.final_performance, result3.insights.len());

    println!("\n📈 Framework Statistics:");
    let report = framework.generate_report();
    println!("{}", report);

    println!("\n🧠 Knowledge Transfer Examples:");
    let general_ml_insights = framework.get_domain_insights(&ResearchDomain::GeneralML);
    println!("General ML insights available: {}", general_ml_insights.len());

    let cv_insights = framework.get_domain_insights(&ResearchDomain::ComputerVision);
    println!("Computer Vision insights available: {}", cv_insights.len());

    println!("\n✨ Research Framework Integration Complete!");
    println!("This example demonstrates how the unified framework");
    println!("successfully consolidates research agents across NAS,");
    println!("HPO, and meta-learning domains with knowledge transfer.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_framework_creation() {
        let config = ResearchConfig::default();
        let framework = UnifiedResearchFramework::new(config);
        assert!(!framework.registry.list_agents().is_empty());
    }

    #[test]
    fn test_workflow_templates() {
        let workflow = WorkflowTemplate::nas_hpo_collaboration("f1_score");
        assert_eq!(workflow.domain, ResearchDomain::AutoML);
        assert!(!workflow.steps.is_empty());
    }

    #[test]
    fn test_agent_registration() {
        let config = ResearchConfig::default();
        let mut framework = UnifiedResearchFramework::new(config);
        framework.registry.register("test_agent", json!({"test": true})).unwrap();
        assert!(framework.registry.has_agent("test_agent"));
    }
}

