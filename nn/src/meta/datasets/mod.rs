//! Meta-Learning Datasets
//!
//! This module provides real-world few-shot learning datasets for meta-learning research,
//! implementing standard benchmarks like Omniglot, miniImageNet, and tieredImageNet.

use coeus_backend::{Backend, DataType, Storage};

pub mod omniglot;
pub mod miniimagenet;
pub mod tiered_imagenet;
pub mod common;

// Re-export main dataset types
pub use omniglot::{OmniglotDataset, OmniglotEpisode};
pub use miniimagenet::{MiniImageNetDataset, MiniImageNetEpisode};
pub use tiered_imagenet::{TieredImageNetDataset, TieredImageNetEpisode};
pub use common::{FewShotEpisode, FewShotDataset, DatasetSplit, DatasetConfig};

/// Standard few-shot learning dataset interface
pub trait MetaDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn is_loaded(&self) -> bool;

    /// Load dataset from disk
    fn load(&mut self, path: &str) -> crate::error::Result<()>;

    /// Get dataset statistics
    fn statistics(&self) -> DatasetStats;

    /// Sample a few-shot episode
    fn sample_episode(
        &self,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        split: DatasetSplit,
    ) -> crate::error::Result<FewShotEpisode<B, S, T>>;
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
    pub image_mean: Vec<f32>, // Normalization mean
    pub image_std: Vec<f32>,  // Normalization std
}
