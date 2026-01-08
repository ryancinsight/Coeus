//! Meta-Learning Datasets
//!
//! This module provides real-world few-shot learning datasets for meta-learning research,
//! implementing standard benchmarks like Omniglot, miniImageNet, and tieredImageNet.

use backend::{Backend, DataType, Storage};

pub mod common;
pub mod miniimagenet;
pub mod omniglot;
pub mod tiered_imagenet;

// Re-export main dataset types
pub use common::{DatasetConfig, DatasetSplit, FewShotDataset, FewShotEpisode};
pub use miniimagenet::{MiniImageNetDataset, MiniImageNetEpisode};
pub use omniglot::{OmniglotDataset, OmniglotEpisode};
pub use tiered_imagenet::{TieredImageNetDataset, TieredImageNetEpisode};

/// Standard few-shot learning dataset interface
pub trait MetaDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn is_loaded(&self) -> bool;

    /// Load dataset from disk
    fn load(&mut self, path: &str) -> crate::core::error::Result<()>;

    /// Get dataset statistics
    fn statistics(&self) -> DatasetStats;

    /// Sample a few-shot episode
    fn sample_episode(
        &self,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        split: DatasetSplit,
    ) -> crate::core::error::Result<FewShotEpisode<B, S, T>>;
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub name: String,
    pub num_classes: usize,
    pub examples_per_class: usize,
    pub train_classes: usize,
    pub val_classes: usize,
    pub test_classes: usize,
    pub total_examples: usize,
    pub image_size: (usize, usize, usize), // (height, width, channels)
    pub image_mean: Vec<f32>,              // Normalization mean
    pub image_std: Vec<f32>,               // Normalization std
}
