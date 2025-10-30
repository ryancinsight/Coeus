//! Common types for few-shot learning datasets

use crate::error::{NNError, Result};
use coeus_backend::{Backend, DataType, Storage};
use coeus_tensor::Tensor;
use rand::Rng;

/// Dataset split for few-shot learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

impl DatasetSplit {
    pub fn name(&self) -> &'static str {
        match self {
            DatasetSplit::Train => "train",
            DatasetSplit::Validation => "validation",
            DatasetSplit::Test => "test",
        }
    }
}

/// Configuration for dataset loading
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Root directory for the dataset
    pub root_dir: String,
    /// Image size (height, width, channels)
    pub image_size: (usize, usize, usize),
    /// Whether to use torchvision-style normalization
    pub normalize: bool,
    /// Data augmentation settings
    pub augment: bool,
    /// Cache loaded data
    pub cache: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            root_dir: "./data".to_string(),
            image_size: (32, 32, 3),
            normalize: true,
            augment: false,
            cache: true,
        }
    }
}

/// Generalized few-shot episode
#[derive(Debug, Clone)]
pub struct FewShotEpisode<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Support set: (image, class_id) pairs
    pub support_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Query set: (image, class_id) pairs
    pub query_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Number of classes in this episode (N-way)
    pub num_classes: usize,
    /// Number of support examples per class (K-shot)
    pub num_support_per_class: usize,
    /// Episode identifier
    pub episode_id: String,
}

/// Generalized few-shot dataset
#[derive(Debug)]
pub struct FewShotDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Dataset configuration
    pub config: DatasetConfig,
    /// Class information (name -> (split, examples))
    pub class_info: std::collections::HashMap<String, (DatasetSplit, Vec<String>)>,
    /// Loaded data cache
    pub data_cache: std::collections::HashMap<String, Tensor<B, S, T>>,
    /// Whether the dataset is fully loaded
    pub loaded: bool,
}

impl<B, S, T> FewShotDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType + From<f32>,
{
    /// Create a new few-shot dataset
    pub fn new(config: DatasetConfig) -> Self {
        Self {
            config,
            class_info: std::collections::HashMap::new(),
            data_cache: std::collections::HashMap::new(),
            loaded: false,
        }
    }

    /// Check if a class belongs to the specified split
    pub fn is_class_in_split(&self, class_name: &str, split: DatasetSplit) -> bool {
        self.class_info
            .get(class_name)
            .is_some_and(|(class_split, _)| *class_split == split)
    }

    /// Get all class names for a split
    pub fn classes_for_split(&self, split: DatasetSplit) -> Vec<String> {
        self.class_info
            .iter()
            .filter_map(|(name, (class_split, _))| {
                if *class_split == split {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Sample N classes from a split
    pub fn sample_classes(&self, n_way: usize, split: DatasetSplit) -> Result<Vec<String>> {
        let available_classes = self.classes_for_split(split);

        if available_classes.len() < n_way {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "Not enough classes in {} split: {} available, {} needed",
                    split.name(),
                    available_classes.len(),
                    n_way
                ),
            });
        }

        // Randomly sample N classes
        let mut selected_classes = Vec::new();
        let mut indices: Vec<usize> = (0..available_classes.len()).collect();
        let mut rng = rand::thread_rng();

        for _ in 0..n_way {
            let idx = rng.gen_range(0..indices.len());
            let selected_idx = indices.swap_remove(idx);
            selected_classes.push(available_classes[selected_idx].clone());
        }

        Ok(selected_classes)
    }

    /// Load example tensor (to be implemented by subclasses)
    pub fn load_example_tensor(&self, _path: &str) -> Result<Tensor<B, S, T>> {
        // Default implementation - should be overridden by subclasses
        // This is a placeholder for actual image loading logic
        Err(NNError::NotImplemented {
            operation: "load_example_tensor".to_string(),
        })
    }

    /// Apply normalization to tensor
    pub fn normalize_tensor(&self, _tensor: &mut Tensor<B, S, T>) {
        if !self.config.normalize {
            return;
        }

        // Default normalization - should be overridden for specific datasets
        // ImageNet-style normalization would be:
        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        let _image_mean = [0.485, 0.456, 0.406];
        let _image_std = [0.229, 0.224, 0.225];

        // Normalize each channel
        // This is a simplified implementation - full normalization would involve:
        // (pixel - mean) / std for each channel
        // For now, just return the tensor as-is
        // TODO: Implement proper tensor normalization
    }

    /// Load all data into cache
    pub fn preload_data(&mut self) -> Result<()> {
        if !self.config.cache {
            return Ok(());
        }

        self.data_cache.clear();

        for (_, examples) in self.class_info.values() {
            for example_path in examples {
                if !self.data_cache.contains_key(example_path) {
                    let tensor = self.load_example_tensor(example_path)?;
                    self.data_cache.insert(example_path.clone(), tensor);
                }
            }
        }

        Ok(())
    }
}
