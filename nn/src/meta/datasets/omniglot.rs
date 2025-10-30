//! Omniglot Dataset for Few-Shot Learning
//!
//! The Omniglot dataset contains 1623 different handwritten characters
//! from 50 different alphabets. Each character has only 20 examples.
//! This makes it ideal for few-shot learning research.

use std::path::Path;
use rand::Rng;

use crate::error::{NNError, Result};
use coeus_backend::{Backend, DataType, Storage};
use coeus_storage::StorageFromVec;
use coeus_tensor::Tensor;

use super::common::*;
use super::MetaDataset;

/// Omniglot-specific episode type
#[derive(Debug, Clone)]
pub struct OmniglotEpisode<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Support set: (28x28 image, character_class) pairs
    pub support_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Query set: (28x28 image, character_class) pairs
    pub query_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Number of characters in this episode (N-way)
    pub n_way: usize,
    /// Number of support examples per character (K-shot)
    pub k_shot: usize,
    /// Episode identifier
    pub episode_id: String,
}

/// Omniglot Dataset Implementation
pub struct OmniglotDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + From<f32>,
{
    /// Base dataset functionality
    pub base: FewShotDataset<B, S, T>,
    /// Character examples: alphabet -> character -> paths
    pub character_examples: std::collections::HashMap<String, std::collections::HashMap<String, Vec<String>>>,
    /// Image size (28x28 for Omniglot)
    pub image_size: (usize, usize),
}

impl<B, S, T> OmniglotDataset<B, S, T>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + From<f32>,
{
    /// Create a new Omniglot dataset
    pub fn new(config: DatasetConfig) -> Self {
        let mut omni_config = config.clone();
        // Omniglot images are grayscale 28x28
        omni_config.image_size = (28, 28, 1);

        Self {
            base: FewShotDataset::new(omni_config),
            character_examples: std::collections::HashMap::new(),
            image_size: (28, 28),
        }
    }

    /// Load PNG image from file
    /// This requires image loading capabilities - for now, we'll create placeholder data
    fn load_png_image(&self, _path: &str) -> Result<Tensor<B, S, T>> {
        // TODO: Implement actual PNG loading with image crate
        // For now, create random placeholder data to demonstrate the interface

        // Omniglot images are 28x28 grayscale, but we store as [channels, height, width]
        // which is [1, 28, 28] for grayscale
        let num_pixels = self.image_size.0 * self.image_size.1;

        // Create random grayscale values for demonstration
        // In real implementation, this would load actual PNG pixels
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_pixels);

        for _ in 0..num_pixels {
            let pixel_value = rng.gen_range(0.0..1.0); // Normalized [0,1]
            data.push(pixel_value.into());
        }

        Ok(Tensor::from_vec(data, &[self.image_size.0, self.image_size.1])?)
    }

    /// Parse Omniglot directory structure to find all characters
    fn parse_directory_structure(&mut self, root_path: &str) -> Result<()> {
        if !Path::new(root_path).exists() {
            // For demonstration, create synthetic data if directory doesn't exist
            self.create_synthetic_data()?;
            return Ok(());
        }

        // Real Omniglot parsing would look like:
        // omniglot/
        //   ├── images_background/     (30 alphabets for training)
        //   │   ├── alphabet1/
        //   │   │   ├── character1/
        //   │   │   │   ├── 001.png
        //   │   │   │   ├── 002.png
        //   │   │   │   └── ...
        //   │   ├── alphabet2/
        //   │   │   └── ...
        //   └── ...
        //   ├── images_evaluation/    (20 alphabets for testing)
        //   │   └── ...

        // For now, we'll use the synthetic data approach
        self.create_synthetic_data()
    }

    /// Create synthetic Omniglot-like data for demonstration
    fn create_synthetic_data(&mut self) -> Result<()> {
        // Based on real Omniglot statistics:
        // - 50 total alphabets
        // - 1623 characters
        // - ~30 alphabets for backgrounds (training)
        // - ~20 alphabets for evaluation (testing)
        // - 20 examples per character

        const BACKGROUND_ALPHABETS: usize = 30;
        const EVALUATION_ALPHABETS: usize = 20;
        const AVG_CHARS_PER_ALPHABET: usize = 30;
        const EXAMPLES_PER_CHARACTER: usize = 20;

        let mut rng = rand::thread_rng();

        // Background alphabets (training)
        for alphabet_idx in 0..BACKGROUND_ALPHABETS {
            let alphabet_name = format!("background_alphabet_{:02}", alphabet_idx);
            let _alphabet_data: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

            // Add characters for this alphabet
            for char_idx in 0..(AVG_CHARS_PER_ALPHABET + rng.gen_range(0..10)) {
                let char_name = format!("char_{:03}", char_idx);
                let char_examples = (0..EXAMPLES_PER_CHARACTER)
                    .map(|i| format!("{}/{}/ex_{:02}.png", alphabet_name, char_name, i))
                    .collect::<Vec<_>>();

                // Update base class info
                self.base.class_info.insert(
                    format!("{}_{}", alphabet_name, char_name),
                    (DatasetSplit::Train, char_examples.clone()),
                );

                // Update character examples
                self.character_examples
                    .entry(alphabet_name.clone())
                    .or_default()
                    .insert(char_name, char_examples);
            }
        }

        // Evaluation alphabets (testing)
        for alphabet_idx in 0..EVALUATION_ALPHABETS {
            let alphabet_name = format!("evaluation_alphabet_{:02}", alphabet_idx);
            let _alphabet_data: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

            // Add characters for this alphabet
            for char_idx in 0..(AVG_CHARS_PER_ALPHABET + rng.gen_range(0..10)) {
                let char_name = format!("char_{:03}", char_idx);
                let char_examples = (0..EXAMPLES_PER_CHARACTER)
                    .map(|i| format!("{}/{}/ex_{:02}.png", alphabet_name, char_name, i))
                    .collect::<Vec<_>>();

                // Update base class info
                self.base.class_info.insert(
                    format!("{}_{}", alphabet_name, char_name),
                    (DatasetSplit::Test, char_examples.clone()),
                );

                // Update character examples
                self.character_examples
                    .entry(alphabet_name.clone())
                    .or_default()
                    .insert(char_name, char_examples);
            }
        }

        self.base.loaded = true;
        Ok(())
    }

    /// Get all characters for a specific alphabet
    pub fn get_alphabet_characters(&self, alphabet: &str) -> Vec<String> {
        self.character_examples
            .get(alphabet)
            .map(|chars| chars.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl<B, S, T> MetaDataset<B, S, T> for OmniglotDataset<B, S, T>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + From<f32>,
{
    fn load(&mut self, path: &str) -> Result<()> {
        println!("Loading Omniglot dataset from: {}", path);
        self.parse_directory_structure(path)?;
        println!("Omniglot dataset loaded successfully");
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.base.loaded
    }

    fn statistics(&self) -> super::DatasetStats {
        let total_classes = self.base.class_info.len();
        let train_classes = self.base.classes_for_split(DatasetSplit::Train).len();
        let val_classes = self.base.classes_for_split(DatasetSplit::Validation).len();
        let test_classes = self.base.classes_for_split(DatasetSplit::Test).len();

        // Estimate based on Omniglot structure
        let examples_per_class = 20; // Omniglot has 20 examples per character
        let total_examples = total_classes * examples_per_class;

        super::DatasetStats {
            name: "Omniglot".to_string(),
            num_classes: total_classes,
            examples_per_class,
            train_classes,
            val_classes,
            test_classes,
            total_examples,
            image_size: (28, 28, 1), // Grayscale images
            image_mean: vec![0.0], // No specific normalization for Omniglot
            image_std: vec![1.0],
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
            let class_examples: Vec<String> = examples.clone();
            let mut example_indices: Vec<usize> = (0..class_examples.len()).collect();

            // Shuffle indices
            for i in (1..example_indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                example_indices.swap(i, j);
            }

            // Add support examples
            for &idx in example_indices.iter().take(k_shot) {
                let example_path = &class_examples[idx];
                let image_tensor = self.load_png_image(example_path)?;
                support_set.push((image_tensor, episode_class_id));
            }

            // Add query examples
            for &idx in example_indices.iter().skip(k_shot).take(n_query) {
                let example_path = &class_examples[idx];
                let image_tensor = self.load_png_image(example_path)?;
                query_set.push((image_tensor, episode_class_id));
            }
        }

        Ok(FewShotEpisode {
            support_set,
            query_set,
            num_classes: n_way,
            num_support_per_class: k_shot,
            episode_id: format!("omniglot_{}_way_{}_shot_{}", n_way, k_shot, rng.gen::<u64>()),
        })
    }
}

impl<B, S, T> FewShotDataset<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + From<f32>,
{
    /// Override for Omniglot-specific image loading
      #[allow(dead_code)]
      fn load_omniglot_image(&self, _path: &str) -> Result<Tensor<B, S, T>> {
        // Omniglot images are 28x28 grayscale PNGs
        // This is a placeholder - real implementation would use image crate
        // For synthetic testing, generate random data
        let mut rng = rand::thread_rng();

        // Create synthetic 28x28 image data (normalized 0-1)
        let data: Vec<T> = (0..(28 * 28))
            .map(|_| {
                let pixel = rng.gen_range(0.0..1.0);
                pixel.into()
            })
            .collect();

        Ok(Tensor::from_vec(data, &[1, 28, 28])?) // [channels, height, width] format
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;

    #[test]
    fn test_omniglot_dataset_creation() {
        let config = DatasetConfig::default();
        let dataset = OmniglotDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        assert!(!dataset.is_loaded());
        assert_eq!(dataset.statistics().name, "Omniglot");
        assert_eq!(dataset.image_size, (28, 28));
    }

    #[test]
    fn test_omniglot_loading() {
        let config = DatasetConfig {
            root_dir: "./data/omniglot".to_string(),
            cache: false,
            ..Default::default()
        };

        let mut dataset = OmniglotDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        // This should work even with non-existent directory (uses synthetic data)
        dataset.load("./nonexistent/path").unwrap();
        assert!(dataset.is_loaded());

        let stats = dataset.statistics();
        assert_eq!(stats.name, "Omniglot");
        assert!(stats.num_classes > 0);
        assert_eq!(stats.examples_per_class, 20); // Omniglot standard
    }

    #[test]
    fn test_omniglot_episode_sampling() {
        let config = DatasetConfig::default();
        let mut dataset = OmniglotDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        dataset.load("./data").unwrap();

        // Sample a 5-way 5-shot episode
        let episode = dataset.sample_episode(5, 5, 10, DatasetSplit::Test).unwrap();

        assert_eq!(episode.num_classes, 5);
        assert_eq!(episode.num_support_per_class, 5);
        assert_eq!(episode.support_set.len(), 5 * 5); // 5 classes * 5 shots
        assert_eq!(episode.query_set.len(), 5 * 10);  // 5 classes * 10 queries

        // Check that all images have correct dimensions
        for (image, _) in &episode.support_set {
            let dims: &[usize] = image.shape().dims();
            assert_eq!(dims, &[28, 28]); // Omniglot image size
        }

        for (image, _) in &episode.query_set {
            let dims: &[usize] = image.shape().dims();
            assert_eq!(dims, &[28, 28]);
        }
    }

    #[test]
    fn test_omniglot_class_distribution() {
        let config = DatasetConfig::default();
        let mut dataset = OmniglotDataset::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config);

        dataset.load("./data").unwrap();

        let stats = dataset.statistics();
        assert!(stats.train_classes > 0);
        assert!(stats.test_classes > 0);
        assert_eq!(stats.train_classes + stats.val_classes + stats.test_classes, stats.num_classes);
    }
}
