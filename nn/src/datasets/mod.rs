//! Vision-Language Dataset Implementations
//!
//! This module provides dataset implementations specifically designed for vision-language tasks,
//! including CLIP-style training with image-text pairs. Supports standard benchmarks like:
//! - COCO (Common Objects in Context)
//! - Flickr30K
//! - Conceptual Captions
//! - Custom image-text datasets
//!
//! ## Features
//! - Memory-efficient batch loading for large datasets (<8GB RAM)
//! - Asynchronous data loading with prefetching
//! - Image-text pair processing and augmentation
//! - Validation and test set preparation
//! - Dataset streaming for distributed training
//!
//! ## Usage
//! ```rust
//! use nn::datasets::{CocoDataset, Flickr30kDataset};
//!
//! // Load COCO dataset
//! let coco_dataset = CocoDataset::new("path/to/coco").await?;
//!
//! // Load Flickr30K dataset
//! let flickr_dataset = Flickr30kDataset::new("path/to/flickr30k").await?;
//!
//! // Create batch loader
//! let loader = VisionLanguageBatchLoader::new(coco_dataset, BatchConfig::default())?;
//! ```

pub mod coco;
pub mod flickr30k;
pub mod vision_language;
pub mod batch_loader;
pub mod transforms;
pub mod streaming;

// Common types for vision-language datasets
#[derive(Debug, Clone)]
pub struct ImageTextPair {
    /// Raw image data as bytes
    pub image_data: Vec<u8>,
    /// Path to image file (optional)
    pub image_path: String,
    /// Associated text captions
    pub captions: Vec<String>,
    /// Unique identifier for this image
    pub image_id: String,
    /// Unique identifiers for each caption
    pub caption_ids: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

// Re-exports
pub use coco::CocoDataset;
pub use flickr30k::Flickr30kDataset;
pub use vision_language::VisionLanguageDataset;
pub use batch_loader::{VisionLanguageBatchLoader, BatchConfig, DatasetBatch};
pub use transforms::Compose;
pub use streaming::StreamingDataset;

// Common types for vision-language datasets
use crate::error::{NNError, Result};
use std::path::Path;
use std::collections::HashMap;
use serde_json;
use tokio::fs;

/// Abstract trait for vision-language datasets
#[async_trait::async_trait(?Send)]
pub trait VisionLanguageData: Send + Sync {
    /// Get the total number of image-text pairs
    fn len(&self) -> usize;

    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get image-text pair by index
    async fn get(&self, index: usize) -> Result<ImageTextPair>;

    /// Get batch of image-text pairs
    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<ImageTextPair>> {
        let mut pairs = Vec::with_capacity(indices.len());
        for &idx in indices {
            pairs.push(self.get(idx).await?);
        }
        Ok(pairs)
    }

    /// Get dataset split (train/val/test)
    fn split(&self) -> DatasetSplit;

    /// Get dataset statistics
    fn statistics(&self) -> DatasetStatistics;
}

/// Dataset splits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
    All,
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    /// Total number of pairs
    pub total_pairs: usize,
    /// Average caption length (words)
    pub avg_caption_length: f64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Image dimensions distribution (if available)
    pub image_sizes: Option<Vec<(u32, u32)>>,
    /// Dataset size on disk (MB)
    pub disk_size_mb: Option<f64>,
}


/// Dataset factory functions
pub mod factory {
    use super::*;

    /// Create a dataset from a common dataset name
    pub async fn create_dataset(name: &str, path: impl AsRef<Path>) -> Result<Box<dyn VisionLanguageData>> {
        match name.to_lowercase().as_str() {
            "coco" | "coco2014" | "coco2017" => {
                Ok(Box::new(CocoDataset::new(&path).await?))
            },
            "flickr30k" | "flickr" => {
                Ok(Box::new(Flickr30kDataset::new(&path).await?))
            },
            _ => {
                // Try to infer from directory structure
                infer_dataset_type(path).await
            }
        }
    }

    /// Infer dataset type from directory structure
    async fn infer_dataset_type(path: impl AsRef<Path>) -> Result<Box<dyn VisionLanguageData>> {
        let path = path.as_ref();

        // Check for COCO-style structure
        if path.join("annotations").exists() && path.join("images").exists() {
            return Ok(Box::new(CocoDataset::new(path).await?));
        }

        // Check for Flickr30K-style structure
        if path.join("flickr30k_images").exists() {
            return Ok(Box::new(Flickr30kDataset::new(path).await?));
        }

        Err(NNError::InvalidInput {
            message: format!("Could not infer dataset type for path: {:?}", path),
        })
    }
}

/// Utility functions for dataset operations
pub mod utils {
    use super::*;

    /// Download and extract dataset if not present
    pub async fn download_dataset(name: &str, target_dir: impl AsRef<Path>) -> Result<()> {
        let target_dir = target_dir.as_ref();

        if target_dir.exists() {
            println!("Dataset {} already exists at {:?}", name, target_dir);
            return Ok(());
        }

        println!("Downloading dataset: {}", name);

        match name.to_lowercase().as_str() {
            "coco" => download_coco(target_dir).await,
            "flickr30k" => download_flickr30k(target_dir).await,
            _ => Err(NNError::InvalidInput {
                message: format!("Unknown dataset: {}. Available: coco, flickr30k", name),
            }),
        }
    }

    async fn download_coco(target_dir: &Path) -> Result<()> {
        println!("COCO dataset download requires manual setup.");
        println!("Please download from: https://cocodataset.org/#download");
        println!("Extract to: {:?}", target_dir);
        println!("Required structure:");
        println!("  {}/", target_dir.display());
        println!("  ├── annotations/");
        println!("  │   ├── captions_train2017.json");
        println!("  │   ├── captions_val2017.json");
        println!("  │   └── instances_train2017.json (optional)");
        println!("  └── images/");
        println!("      ├── train2017/");
        println!("      └── val2017/");

        Err(NNError::InvalidInput {
            message: "COCO dataset requires manual download and setup".to_string(),
        })
    }

    async fn download_flickr30k(target_dir: &Path) -> Result<()> {
        println!("Flickr30K dataset download requires manual setup.");
        println!("Please download from: http://shannon.cs.illinois.edu/DenotationGraph/");
        println!("Extract to: {:?}", target_dir);

        Err(NNError::InvalidInput {
            message: "Flickr30K dataset requires manual download and setup".to_string(),
        })
    }

    /// Verify dataset integrity
    pub async fn verify_dataset(dataset: &dyn VisionLanguageData) -> Result<DatasetVerification> {
        println!("Verifying dataset with {} pairs...", dataset.len());

        let mut stats = DatasetVerification {
            total_pairs: dataset.len(),
            valid_pairs: 0,
            corrupt_images: 0,
            missing_captions: 0,
            sample_caption_lengths: Vec::new(),
        };

        // Sample a subset for verification (first 1000 or all if smaller)
        let sample_size = std::cmp::min(1000, dataset.len());
        let step = dataset.len() / sample_size;

        for i in (0..dataset.len()).step_by(step).take(sample_size) {
            match dataset.get(i).await {
                Ok(pair) => {
                    if pair.image_data.is_empty() {
                        stats.corrupt_images += 1;
                    } else {
                        stats.valid_pairs += 1;
                    }

                    if pair.captions.is_empty() {
                        stats.missing_captions += 1;
                    } else {
                        for caption in &pair.captions {
                            stats.sample_caption_lengths.push(caption.split_whitespace().count());
                        }
                    }
                },
                Err(_) => {
                    stats.corrupt_images += 1;
                }
            }
        }

        Ok(stats)
    }

    #[derive(Debug)]
    pub struct DatasetVerification {
        pub total_pairs: usize,
        pub valid_pairs: usize,
        pub corrupt_images: usize,
        pub missing_captions: usize,
        pub sample_caption_lengths: Vec<usize>,
    }

    impl DatasetVerification {
        pub fn print_report(&self) {
            println!("=== Dataset Verification Report ===");
            println!("Total Pairs: {}", self.total_pairs);
            println!("Valid Pairs: {} ({:.1}%)", self.valid_pairs, (self.valid_pairs as f64 / self.total_pairs as f64) * 100.0);
            println!("Corrupt Images: {}", self.corrupt_images);
            println!("Missing Captions: {}", self.missing_captions);

            if !self.sample_caption_lengths.is_empty() {
                let avg_len = self.sample_caption_lengths.iter().sum::<usize>() as f64 / self.sample_caption_lengths.len() as f64;
                println!("Avg Caption Length: {:.1} words", avg_len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dataset_factory() {
        // Test with invalid path should return error
        let result = factory::create_dataset("coco", "/nonexistent/path").await;
        assert!(result.is_err());
    }
}
