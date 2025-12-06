//! COCO (Common Objects in Context) Dataset Implementation
//!
//! This module provides a complete implementation for loading the COCO dataset
//! for vision-language training. COCO contains ~118K training images and ~5K
//! validation images, each with multiple captions.
//!
//! ## Dataset Structure
//! The expected directory structure is:
//! ```
//! coco/
//! ├── annotations/
//! │   ├── captions_train2017.json
//! │   ├── captions_val2017.json
//! │   └── instances_train2017.json (optional)
//! └── images/
//!     ├── train2017/
//!     └── val2017/
//! ```
//!
//! ## Features
//! - Asynchronous image loading and caption parsing
//! - Memory-efficient storage of annotations
//! - Train/validation split support
//! - Caption preprocessing and filtering
//! - Image aspect ratio handling for batch formation

use crate::error::{NNError, Result};
use super::{VisionLanguageData, VisionLanguageDataset, ImageTextPair, DatasetSplit, DatasetStatistics};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;
use futures::prelude::*;

/// COCO dataset implementation
pub struct CocoDataset {
    /// Base dataset directory
    base_path: PathBuf,
    /// Parsed annotations (image_id -> captions)
    annotations: HashMap<String, Vec<String>>,
    /// Image metadata (filename -> image_id)
    image_metadata: Vec<ImageMetadata>,
    /// Dataset split
    split: DatasetSplit,
    /// Image base directory for this split
    image_dir: PathBuf,
    /// Statistics
    statistics: DatasetStatistics,
}

/// COCO annotation entry (from captions JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Annotation {
    /// Unique caption identifier
    id: u64,
    /// Associated image identifier
    image_id: u64,
    /// Textual caption
    caption: String,
}

/// COCO image entry (from images JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageInfo {
    /// Unique image identifier
    id: u64,
    /// Width in pixels
    width: u32,
    /// Height in pixels
    height: u32,
    /// Image filename (e.g., "00000000025.jpg")
    file_name: String,
    /// Image aspect ratio (width/height)
    aspect_ratio: Option<f64>,
}

/// COCO annotations JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoAnnotations {
    /// List of all annotations
    annotations: Vec<Annotation>,
    /// List of all images
    images: Vec<ImageInfo>,
    /// List of all categories (optional)
    categories: Option<Vec<Category>>,
}

/// COCO category information (optional)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Category {
    id: u64,
    name: String,
    supercategory: String,
}

/// Image metadata for efficient access
#[derive(Debug, Clone)]
struct ImageMetadata {
    /// Image ID (as string)
    image_id: String,
    /// Image filename
    filename: String,
    /// Image dimensions
    dimensions: (u32, u32),
    /// Pre-computed captions
    captions: Vec<String>,
    /// Image aspect ratio
    aspect_ratio: f64,
    /// Full image path
    image_path: PathBuf,
}

impl CocoDataset {
    /// Create a new COCO dataset loader
    ///
    /// # Arguments
    /// * `base_path` - Path to the COCO dataset directory
    ///
    /// # Returns
    /// * `CocoDataset` - The dataset loader for the training split
    ///
    /// # Example
    /// ```rust
    /// let dataset = CocoDataset::new("path/to/coco").await?;
    /// println!("Loaded dataset with {} images", dataset.len());
    /// ```
    pub async fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_split(base_path, DatasetSplit::Train).await
    }

    /// Create a new COCO dataset loader for a specific split
    ///
    /// # Arguments
    /// * `base_path` - Path to the COCO dataset directory
    /// * `split` - Dataset split to load (Train, Validation, or All)
    pub async fn with_split(base_path: impl AsRef<Path>, split: DatasetSplit) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Validate directory structure
        Self::validate_directory_structure(&base_path)?;

        // Load annotations based on split
        let (annotations_file, image_subdir) = match split {
            DatasetSplit::Train => ("captions_train2017.json", "train2017"),
            DatasetSplit::Validation => ("captions_val2017.json", "val2017"),
            DatasetSplit::Test => return Err(NNError::InvalidInput {
                message: "COCO test split requires separate test2017 dataset".to_string(),
            }),
            DatasetSplit::All => return Err(NNError::InvalidInput {
                message: "Use Train or Validation split specifically for COCO".to_string(),
            }),
        };

        let annotations_path = base_path.join("annotations").join(annotations_file);
        let image_dir = base_path.join("images").join(image_subdir);

        // Load and parse annotations
        let coco_data = Self::load_annotations(&annotations_path).await?;

        // Build metadata and annotations mapping
        let (image_metadata, annotations, statistics) = Self::build_metadata(&coco_data, &image_dir)?;

        Ok(Self {
            base_path,
            annotations,
            image_metadata,
            split,
            image_dir,
            statistics,
        })
    }

    /// Validate the expected directory structure exists
    fn validate_directory_structure(base_path: &Path) -> Result<()> {
        let annotations_dir = base_path.join("annotations");
        let images_dir = base_path.join("images");

        if !annotations_dir.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Annotations directory not found: {:?}", annotations_dir),
            });
        }

        if !images_dir.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Images directory not found: {:?}", images_dir),
            });
        }

        // Check for required annotation files
        let train_annotations = annotations_dir.join("captions_train2017.json");
        let val_annotations = annotations_dir.join("captions_val2017.json");

        if !train_annotations.exists() && !val_annotations.exists() {
            return Err(NNError::InvalidInput {
                message: "Neither train nor validation captions found. Download from: https://cocodataset.org/#download".to_string(),
            });
        }

        Ok(())
    }

    /// Load and parse COCO annotations from JSON file
    async fn load_annotations(annotations_path: &Path) -> Result<CocoAnnotations> {
        println!("Loading COCO annotations from: {:?}", annotations_path);

        if !annotations_path.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Annotations file not found: {:?}", annotations_path),
            });
        }

        let content = fs::read_to_string(annotations_path).await?;

        let coco_data: CocoAnnotations = serde_json::from_str(&content)
            .map_err(|e| NNError::InvalidInput {
                message: format!("Failed to parse COCO annotations JSON: {}", e),
            })?;

        println!("Loaded {} annotations for {} images",
                coco_data.annotations.len(), coco_data.images.len());

        Ok(coco_data)
    }

    /// Build metadata structures from parsed COCO data
    fn build_metadata(
        coco_data: &CocoAnnotations,
        image_dir: &Path,
    ) -> Result<(Vec<ImageMetadata>, HashMap<String, Vec<String>>, DatasetStatistics)> {
        let mut image_map = HashMap::new();
        let mut image_metadata = Vec::new();
        let mut annotations = HashMap::new();
        let mut image_sizes = Vec::new();

        // Build image map for quick lookup
        for image in &coco_data.images {
            let image_id_str = image.id.to_string();
            let filename = format!("{:012}.jpg", image.id);

            image_map.insert(image.id, ImageMetadata {
                image_id: image_id_str.clone(),
                filename: filename.clone(),
                dimensions: (image.width, image.height),
                captions: Vec::new(), // Will be filled below
                aspect_ratio: image.width as f64 / image.height as f64,
                image_path: image_dir.join(filename),
            });

            image_sizes.push((image.width, image.height));
        }

        // Group captions by image
        for annotation in &coco_data.annotations {
            let image_id_str = annotation.image_id.to_string();

            annotations
                .entry(image_id_str.clone())
                .or_insert_with(Vec::new)
                .push(annotation.caption.clone());
        }

        // Build final metadata with captions
        for annotation in &coco_data.annotations {
            if let Some(metadata) = image_map.get_mut(&annotation.image_id) {
                // Only add if not already present (handle duplicate annotations)
                if !metadata.captions.contains(&annotation.caption) {
                    metadata.captions.push(annotation.caption.clone());
                }
            }
        }

        // Convert to vector and compute statistics
        image_metadata.extend(image_map.into_values());

        let total_captions: usize = image_metadata.iter().map(|m| m.captions.len()).sum();
        let avg_caption_length = Self::compute_average_caption_length(&image_metadata);

        let statistics = DatasetStatistics {
            total_pairs: total_captions,
            avg_caption_length,
            vocab_size: 0, // Would need to compute from actual tokenization
            image_sizes: Some(image_sizes),
            disk_size_mb: Self::estimate_disk_size(&image_metadata, image_dir),
        };

        Ok((image_metadata, annotations, statistics))
    }

    /// Compute average caption length across all captions
    fn compute_average_caption_length(metadata: &[ImageMetadata]) -> f64 {
        let mut total_words = 0;
        let mut total_captions = 0;

        for image_meta in metadata {
            for caption in &image_meta.captions {
                total_words += caption.split_whitespace().count();
                total_captions += 1;
            }
        }

        if total_captions == 0 {
            0.0
        } else {
            total_words as f64 / total_captions as f64
        }
    }

    /// Estimate dataset size on disk
    fn estimate_disk_size(metadata: &[ImageMetadata], image_dir: &Path) -> Option<f64> {
        // Rough estimation based on image count (average JPEG size ~200KB)
        let avg_image_size_mb = 0.2;
        let total_images = metadata.len();
        Some(total_images as f64 * avg_image_size_mb)
    }

    /// Filter captions by quality (remove too short/long captions)
    pub fn filter_captions(&mut self, min_words: usize, max_words: usize) {
        println!("Filtering captions: {}-{} words", min_words, max_words);

        let mut filtered_annotations = HashMap::new();
        let mut removed_count = 0;

        for (image_id, captions) in &self.annotations {
            let filtered: Vec<String> = captions
                .iter()
                .filter(|caption| {
                    let word_count = caption.split_whitespace().count();
                    word_count >= min_words && word_count <= max_words
                })
                .cloned()
                .collect();

            if !filtered.is_empty() {
                filtered_annotations.insert(image_id.clone(), filtered);
            } else {
                removed_count += 1;
            }
        }

        self.annotations = filtered_annotations;
        println!("Removed {} images with no valid captions", removed_count);

        // Update image metadata
        self.image_metadata = self.image_metadata
            .iter()
            .filter(|meta| self.annotations.contains_key(&meta.image_id))
            .cloned()
            .collect();

        // Update statistics
        self.statistics.total_pairs = self.annotations.values().map(|v| v.len()).sum();
        self.statistics.avg_caption_length = Self::compute_average_caption_length(&self.image_metadata);
    }
}

#[async_trait::async_trait(?Send)]
impl VisionLanguageData for CocoDataset {
    fn len(&self) -> usize {
        self.annotations.values().map(|captions| captions.len()).sum()
    }

    fn get(&self, mut index: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ImageTextPair>> + Send + '_>> {
        let image_metadata = self.image_metadata.clone();
        let split = self.split.clone();
        let total_len = self.len();

        Box::pin(async move {
            // Find which image this global index refers to
            for metadata in &image_metadata {
                let caption_count = metadata.captions.len();
                if index < caption_count {
                    // Load image data
                    let image_data = Self::load_image_data_static(&metadata.image_path).await?;

                    return Ok(ImageTextPair {
                        image_data,
                        image_path: metadata.image_path.to_string_lossy().to_string(),
                        captions: vec![metadata.captions[index].clone()],
                        image_id: metadata.image_id.clone(),
                        caption_ids: vec![format!("{}_{}", metadata.image_id, index)],
                        metadata: HashMap::from([
                            ("width".to_string(), serde_json::json!(metadata.dimensions.0.to_string())),
                            ("height".to_string(), serde_json::json!(metadata.dimensions.1.to_string())),
                            ("aspect_ratio".to_string(), serde_json::json!(metadata.aspect_ratio.to_string())),
                            ("split".to_string(), serde_json::json!(format!("{:?}", split).to_lowercase())),
                        ]),
                    });
                }
                index -= caption_count;
            }

            Err(NNError::InvalidInput {
                message: format!("Index {} out of bounds for {} pairs", index + 1, total_len),
            })
        })
    }

    fn split(&self) -> DatasetSplit {
        self.split
    }

    fn statistics(&self) -> DatasetStatistics {
        self.statistics.clone()
    }
}

impl CocoDataset {
    /// Load image data from file
    async fn load_image_data(&self, image_path: &Path) -> Result<Vec<u8>> {
        Self::load_image_data_static(image_path).await
    }

    async fn load_image_data_static(image_path: &Path) -> Result<Vec<u8>> {
        if !image_path.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Image file not found: {:?}", image_path),
            });
        }

        fs::read(image_path).await.map_err(|e| NNError::InvalidInput {
            message: format!("Failed to read image file: {}", e),
        })
    }

    /// Get all images for a specific image ID
    pub fn get_images_by_id(&self, image_id: &str) -> Option<&ImageMetadata> {
        self.image_metadata.iter()
            .find(|meta| meta.image_id == image_id)
    }

    /// Get captions for a specific image ID
    pub fn get_captions_by_id(&self, image_id: &str) -> Option<&Vec<String>> {
        self.annotations.get(image_id)
    }

    /// Get dataset in wrapped VisionLanguageDataset format
    pub fn as_vision_language_dataset(self) -> VisionLanguageDataset<Self> {
        VisionLanguageDataset::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs;
    use std::fs as std_fs;

    // Create minimal test JSON files
    async fn create_test_annotations(dir: &Path) -> Result<()> {
        let annotations = CocoAnnotations {
            annotations: vec![
                Annotation {
                    id: 1,
                    image_id: 1,
                    caption: "A test image with a cat".to_string(),
                },
                Annotation {
                    id: 2,
                    image_id: 1,
                    caption: "A cat sitting on a table".to_string(),
                },
                Annotation {
                    id: 3,
                    image_id: 2,
                    caption: "A dog in the park".to_string(),
                },
            ],
            images: vec![
                ImageInfo {
                    id: 1,
                    width: 640,
                    height: 480,
                    file_name: "000000000001.jpg".to_string(),
                    aspect_ratio: Some(640.0 / 480.0),
                },
                ImageInfo {
                    id: 2,
                    width: 800,
                    height: 600,
                    file_name: "000000000002.jpg".to_string(),
                    aspect_ratio: Some(800.0 / 600.0),
                },
            ],
            categories: None,
        };

        // Create annotations directory
        let annotations_dir = dir.join("annotations");
        std_fs::create_dir_all(&annotations_dir)?;

        // Write annotations file
        let annotations_path = annotations_dir.join("captions_train2017.json");
        let content = serde_json::to_string(&annotations)?;
        fs::write(annotations_path, content).await?;

        // Create images directory and fake image files
        let images_dir = dir.join("images").join("train2017");
        std_fs::create_dir_all(&images_dir)?;

        // Create dummy image files
        fs::write(images_dir.join("000000000001.jpg"), b"fake_jpeg_data_1").await?;
        fs::write(images_dir.join("000000000002.jpg"), b"fake_jpeg_data_2").await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_coco_dataset_loading() {
        let temp_dir = tempdir().unwrap();
        let coco_path = temp_dir.path();

        create_test_annotations(coco_path).await.unwrap();

        let dataset = CocoDataset::with_split(coco_path, DatasetSplit::Train).await.unwrap();

        assert_eq!(dataset.len(), 3); // 3 captions total
        assert_eq!(dataset.split(), DatasetSplit::Train);

        // Test getting first item
        let pair = dataset.get(0).await.unwrap();
        assert_eq!(pair.captions[0], "A test image with a cat");
        assert_eq!(pair.image_id, "1");
        assert!(!pair.image_data.is_empty());
    }

    #[tokio::test]
    async fn test_caption_filtering() {
        let temp_dir = tempdir().unwrap();
        let coco_path = temp_dir.path();

        create_test_annotations(coco_path).await.unwrap();

        let mut dataset = CocoDataset::with_split(coco_path, DatasetSplit::Train).await.unwrap();

        // Filter to only 5+ word captions (should keep 2, remove 1)
        dataset.filter_captions(5, 10);
        assert_eq!(dataset.len(), 2);
    }

    #[tokio::test]
    async fn test_nonexistent_path() {
        let result = CocoDataset::new("/nonexistent/path").await;
        assert!(result.is_err());
    }
}
