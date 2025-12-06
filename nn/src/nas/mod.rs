//! Neural Architecture Search (NAS) automation.
//!
//! This module provides automated neural architecture search capabilities
//! including evolutionary algorithms, reinforcement learning-based search,
//! and differentiable architecture search (DARTS).

pub mod darts;
pub mod evaluator;
pub mod evolutionary;
pub mod reinforcement;
pub mod research_automation;
pub mod search_space;
pub mod utils;

// Re-export research automation types
pub use research_automation::{
    AutomatedResearchPipeline, ExperimentRecord, KnowledgeBase, NasResearchDomain, ResearchHypothesis,
    ResearchInsight, ResearchMetrics, ResearchPipelineConfig, ResearchState, StatisticalTest,
};
// Re-export main NAS types
pub use darts::DartsNAS;
pub use evaluator::{ArchitectureEvaluator, SimpleEvaluator};
pub use evolutionary::EvolutionaryNAS;
pub use reinforcement::ReinforcementNAS;
pub use search_space::{Architecture, ArchitectureSpace};
