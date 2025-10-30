//! MiniImageNet Dataset for Few-Shot Learning
//!
//! MiniImageNet is a subset of ImageNet containing 100 classes,
//! commonly used for few-shot learning research. Each class has
//! 600 examples (84x84 RGB images).

use crate::error::{NNError, Result};
use coeus_backend::{Backend, DataType, Storage};
use coeus_dtype::traits::FloatExt;
use coeus_storage::StorageFromVec;
use rand::Rng;
use coeus_tensor::Tensor;

use super::common::*;
use super::MetaDataset;

/// MiniImageNet-specific episode type
#[derive(Debug, Clone)]
pub struct MiniImageNetEpisode<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Support set: (84x84 RGB image, class_id) pairs
    pub support_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Query set: (84x84 RGB image, class_id) pairs
    pub query_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Number of classes in this episode (N-way)
    pub n_way: usize,
    /// Number of support examples per class (K-shot)
    pub k_shot: usize,
    /// Episode identifier
    pub episode_id: String,
}

/// MiniImageNet Dataset Implementation
pub struct MiniImageNetDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt + From<f32>,
{
    /// Base dataset functionality
    pub base: FewShotDataset<B, S, T>,
    /// Class examples: class_name -> example_paths
    pub class_examples: std::collections::HashMap<String, Vec<String>>,
    /// Image size (84x84 for MiniImageNet)
    pub image_size: (usize, usize),
}

impl<B, S, T> MiniImageNetDataset<B, S, T>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt + From<f32>,
{
    /// Create a new MiniImageNet dataset
    pub fn new(config: DatasetConfig) -> Self {
        let mut mini_config = config.clone();
        // MiniImageNet images are 84x84 RGB
        mini_config.image_size = (84, 84, 3);

        Self {
            base: FewShotDataset::new(mini_config),
            class_examples: std::collections::HashMap::new(),
            image_size: (84, 84),
        }
    }

    /// Load JPEG image from file (placeholder implementation)
    fn load_jpeg_image(&self, _path: &str) -> Result<Tensor<B, S, T>> {
        // TODO: Implement actual JPEG loading with image crate
        // For now, create random RGB data to demonstrate the interface

        // MiniImageNet images are 84x84 RGB
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(84 * 84 * 3);

        for _ in 0..(84 * 84 * 3) {
            let pixel = rng.gen_range(0.0..1.0); // Normalized [0,1]
            data.push(pixel.into());
        }

        // Store as [channels, height, width] = [3, 84, 84]
        Tensor::from_vec(data, &[3, 84, 84]).map_err(|e| {
            NNError::InvalidInput {
                message: format!("Failed to create tensor from image data: {}", e),
            }
        })
    }

    /// Create synthetic MiniImageNet-like data for demonstration
    fn create_synthetic_data(&mut self) -> Result<()> {
        // Standard MiniImageNet has 100 classes, ~600 examples each
        #[allow(dead_code)]
        const NUM_CLASSES: usize = 100;
        const EXAMPLES_PER_CLASS: usize = 600;
        let _rng = rand::thread_rng();

        // Standard MiniImageNet splits: 64 train, 16 val, 20 test
        const TRAIN_CLASSES: usize = 64;
        const VAL_CLASSES: usize = 16;
        const TEST_CLASSES: usize = 20;

        // Create training classes
        for i in 0..TRAIN_CLASSES {
            let class_name = format!("train_class_{:03}", i);
            let examples = (0..EXAMPLES_PER_CLASS)
                .map(|j| format!("{}/img_{:04}.jpg", class_name, j))
                .collect::<Vec<_>>();

            self.class_examples.insert(class_name.clone(), examples.clone());
            self.base.class_info.insert(class_name, (DatasetSplit::Train, examples));
        }

        // Create validation classes
        for i in 0..VAL_CLASSES {
            let class_name = format!("val_class_{:03}", i);
            let examples = (0..EXAMPLES_PER_CLASS)
                .map(|j| format!("{}/img_{:04}.jpg", class_name, j))
                .collect::<Vec<_>>();

            self.class_examples.insert(class_name.clone(), examples.clone());
            self.base.class_info.insert(class_name, (DatasetSplit::Validation, examples));
        }

        // Create test classes
        for i in 0..TEST_CLASSES {
            let class_name = format!("test_class_{:03}", i);
            let examples = (0..EXAMPLES_PER_CLASS)
                .map(|j| format!("{}/img_{:04}.jpg", class_name, j))
                .collect::<Vec<_>>();

            self.class_examples.insert(class_name.clone(), examples.clone());
            self.base.class_info.insert(class_name, (DatasetSplit::Test, examples));
        }

        self.base.loaded = true;
        Ok(())
    }
}

impl<B, S, T> MetaDataset<B, S, T> for MiniImageNetDataset<B, S, T>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt + From<f32>,
{
    fn load(&mut self, path: &str) -> Result<()> {
        println!("Loading MiniImageNet dataset from: {}", path);
        // For now, always use synthetic data
        // TODO: Implement real data loading when actual dataset is available
        self.create_synthetic_data()?;
        println!("MiniImageNet dataset loaded successfully");
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.base.loaded
    }

    fn statistics(&self) -> super::DatasetStats {
        super::DatasetStats {
            name: "MiniImageNet".to_string(),
            num_classes: 100,
            examples_per_class: 600,
            train_classes: 64,
            val_classes: 16,
            test_classes: 20,
            total_examples: 100 * 600,
            image_size: (84, 84, 3), // RGB images
            image_mean: vec![0.485, 0.456, 0.406], // ImageNet means
            image_std: vec![0.229, 0.224, 0.225],  // ImageNet stds
        }
    }

    fn sample_episode(
        &self,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        split: DatasetSplit,
    ) -> Result<FewShotEpisode<B, S, T>> {
        if !self.is_loaded() {
            return Err(NNError::InvalidConfiguration {
                message: "Dataset not loaded. Call load() first.".to_string(),
            });
        }

        let selected_classes = self.base.sample_classes(n_way, split)?;

        let mut support_set = Vec::new();
        let mut query_set = Vec::new();
        let mut rng = rand::thread_rng();

        // For each selected class, sample K-shot + N-query examples
        for (episode_class_id, class_name) in selected_classes.iter().enumerate() {
            let (_, examples) = self.base.class_info.get(class_name).unwrap();

            if examples.len() < k_shot + n_query {
                return Err(NNError::InvalidConfiguration {
                    message: format!(
                        "Class {} has insufficient examples: {} available, {} needed",
                        class_name,
                        examples.len(),
                        k_shot + n_query
                    ),
                });
            }

            // Sample examples for this class
            let mut example_indices: Vec<usize> = (0..examples.len()).collect();

            // Shuffle indices
            for i in (1..example_indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                example_indices.swap(i, j);
            }

            // Add support examples
            for &idx in example_indices.iter().take(k_shot) {
                let example_path = &examples[idx];
                let image_tensor = self.load_jpeg_image(example_path)?;
                support_set.push((image_tensor, episode_class_id));
            }

            // Add query examples
            for &idx in example_indices.iter().skip(k_shot).take(n_query) {
                let example_path = &examples[idx];
                let image_tensor = self.load_jpeg_image(example_path)?;
                query_set.push((image_tensor, episode_class_id));
            }
        }

        Ok(FewShotEpisode {
            support_set,
            query_set,
            num_classes: n_way,
            num_support_per_class: k_shot,
            episode_id: format!("miniimagenet_{}_way_{}_shot_{}", n_way, k_shot, rng.gen::<u64>()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;

    #[test]
    fn test_miniimagenet_dataset_creation() {
        let config = DatasetConfig::default();
        let dataset = MiniImageNetDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        assert!(!dataset.is_loaded());
        assert_eq!(dataset.image_size, (84, 84));
    }

    #[test]
    fn test_miniimagenet_loading() {
        let config = DatasetConfig::default();
        let mut dataset = MiniImageNetDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        dataset.load("./data").unwrap();
        assert!(dataset.is_loaded());

        let stats = dataset.statistics();
        assert_eq!(stats.name, "MiniImageNet");
        assert_eq!(stats.num_classes, 100);
        assert_eq!(stats.examples_per_class, 600);
    }
}
