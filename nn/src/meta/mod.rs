//! Meta-Learning Systems.
//!
//! This module provides comprehensive meta-learning capabilities including
//! learning-to-learn algorithms, few-shot learning, and adaptation strategies.

pub mod adaptation;
#[cfg(feature = "research")]
pub mod adapters;
pub mod benchmark;
pub mod datasets;
pub mod maml;
pub mod meta_optimizer;
pub mod prototypical;
pub mod task_distribution;

// Re-export main meta-learning types
pub use adaptation::MetaLearner;
#[cfg(feature = "research")]
pub use adapters::{MAMLAdapter, MAMLAgentFactory, PrototypicalAdapter, PrototypicalAgentFactory};
pub use benchmark::{FewShotDataset, MetaLearningBenchmark};
pub use datasets::{DatasetSplit, DatasetStats, FewShotEpisode, MetaDataset};
// Re-export dataset types
pub use maml::MAML;
pub use meta_optimizer::{MetaLSTM, MetaSGD};
pub use prototypical::PrototypicalNetwork;
pub use task_distribution::TaskDistribution;
