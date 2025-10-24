//! Automated Feature Engineering.
//!
//! This module provides comprehensive automated feature engineering capabilities
//! including feature selection, transformation, pipeline construction, and
//! integration with NAS, HPO, and meta-learning.

pub mod importance;
pub mod integration;
pub mod pipeline;
pub mod selection;
pub mod transformation;

// Re-export main feature engineering types
pub use importance::{FeatureImportance, ImportanceMethod, ImportanceResult};
pub use integration::{AutoFeatureEngineer, FeatureEngineeringResult};
pub use pipeline::{FeaturePipeline, PipelineResult, PipelineStep};
pub use selection::{FeatureSelectionResult, FeatureSelector, SelectionMethod};
pub use transformation::{FeatureTransformationResult, FeatureTransformer, TransformationMethod};
