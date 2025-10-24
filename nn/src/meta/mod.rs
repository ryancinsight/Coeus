//! Meta-Learning Systems.
//!
//! This module provides comprehensive meta-learning capabilities including
//! learning-to-learn algorithms, few-shot learning, and adaptation strategies.

pub mod adaptation;
pub mod benchmark;
pub mod maml;
pub mod meta_optimizer;
pub mod prototypical;
pub mod task_distribution;

// Re-export main meta-learning types
pub use adaptation::MetaLearner;
pub use benchmark::{FewShotDataset, MetaLearningBenchmark};
pub use maml::MAML;
pub use meta_optimizer::{MetaLSTM, MetaSGD};
pub use prototypical::PrototypicalNetwork;
pub use task_distribution::TaskDistribution;
