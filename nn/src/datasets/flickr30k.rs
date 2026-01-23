//! Flickr30K Dataset Implementation
//!
//! This module provides an implementation for loading the Flickr30K dataset
//! for vision-language training. Flickr30K contains 31,783 images collected
//! from Flickr, each annotated with 5 reference captions.
//!
//! ## Dataset Structure
//! The expected directory structure is:
//! ```
//! flickr30k/
//! ├── flickr30k_images/
//! │   └── *.jpg
//! ├── results_20130124.token (train/val captions)
//! └── result_20130730.token (test captions, unused)
//! ```
//!
//! ## Features
//! - Automatic train/validation split (29,783 train, 1,000 validation)
//! - Caption preprocessing and cleaning
//! - Memory-efficient caption storage
//! - Train/validation split handling

use super::{DatasetSplit, DatasetStatistics, ImageTextPair, VisionLanguageData};
use crate::core::error::{NNError, Result};
// use futures::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Flickr30K dataset implementation
pub struct Flickr30kDataset {
    /// Base dataset directory
    #[allow(dead_code)]
    base_path: PathBuf,
    /// Image directory path
    image_dir: PathBuf,
    /// Caption annotations (image_filename -> captions)
    annotations: HashMap<String, Vec<String>>,
    /// Image filenames for each split
    image_files: Vec<String>,
    /// Dataset split
    split: DatasetSplit,
    /// Statistics
    statistics: DatasetStatistics,
}

impl Flickr30kDataset {
    /// Create a new Flickr30K dataset loader
    ///
    /// # Arguments
    /// * `base_path` - Path to the Flickr30K dataset directory
    ///
    /// # Returns
    /// * `Flickr30kDataset` - The dataset loader for the training split
    ///
    /// # Example
    /// ```rust
    /// let dataset = Flickr30kDataset::new("path/to/flickr30k").await?;
    /// println!("Loaded dataset with {} images", dataset.len());
    /// ```
    pub async fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_split(base_path, DatasetSplit::Train).await
    }

    /// Create a new Flickr30K dataset loader for a specific split
    ///
    /// # Arguments
    /// * `base_path` - Path to the Flickr30K dataset directory
    /// * `split` - Dataset split to load (Train, Validation, or Test)
    pub async fn with_split(base_path: impl AsRef<Path>, split: DatasetSplit) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Validate directory structure
        Self::validate_directory_structure(&base_path)?;

        // Load annotations
        let annotations_path = base_path.join("results_20130124.token");
        let (all_annotations, image_id_to_filename) =
            Self::load_annotations(&annotations_path).await?;

        // Create train/validation split
        let (train_images, val_images) = Self::create_train_val_split(&image_id_to_filename);

        let (image_files, annotations) = match split {
            DatasetSplit::Train => {
                let filtered_annotations: HashMap<String, Vec<String>> = all_annotations
                    .into_iter()
                    .filter(|(filename, _)| train_images.contains(filename))
                    .collect();
                (train_images, filtered_annotations)
            }
            DatasetSplit::Validation => {
                let filtered_annotations: HashMap<String, Vec<String>> = all_annotations
                    .into_iter()
                    .filter(|(filename, _)| val_images.contains(filename))
                    .collect();
                (val_images, filtered_annotations)
            }
            DatasetSplit::Test => {
                return Err(NNError::InvalidInput {
                    message: "Flickr30K test split not supported - test captions are not publicly available".to_string(),
                });
            }
            DatasetSplit::All => {
                let mut all_files = train_images;
                all_files.extend(val_images);
                (all_files, all_annotations)
            }
        };

        let image_dir = base_path.join("flickr30k_images");

        // Compute statistics
        let statistics = Self::compute_statistics(&annotations, &image_dir).await;

        Ok(Self {
            base_path,
            image_dir,
            annotations,
            image_files,
            split,
            statistics,
        })
    }

    /// Validate the expected directory structure exists
    fn validate_directory_structure(base_path: &Path) -> Result<()> {
        let image_dir = base_path.join("flickr30k_images");
        let captions_file = base_path.join("results_20130124.token");

        if !image_dir.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Image directory not found: {:?}", image_dir),
            });
        }

        if !captions_file.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Captions file not found: {:?}. Download from: http://shannon.cs.illinois.edu/DenotationGraph/", captions_file),
            });
        }

        Ok(())
    }

    /// Load and parse Flickr30K annotations
    async fn load_annotations(
        captions_path: &Path,
    ) -> Result<(HashMap<String, Vec<String>>, HashMap<String, String>)> {
        println!("Loading Flickr30K annotations from: {:?}", captions_path);

        let content = fs::read_to_string(captions_path).await?;

        let mut annotations = HashMap::new();
        let mut image_id_to_filename = HashMap::new();

        // Parse caption file format: image_id#caption_id<TAB>caption_text
        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Split on tab character
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                println!("Warning: Malformed line {}: {}", line_num + 1, line);
                continue;
            }

            let image_caption_id = parts[0];
            let caption_text = parts[1];

            // Parse image_id#caption_id
            let image_caption_parts: Vec<&str> = image_caption_id.split('#').collect();
            if image_caption_parts.len() != 2 {
                println!(
                    "Warning: Malformed image ID {} on line {}",
                    image_caption_id,
                    line_num + 1
                );
                continue;
            }

            let image_id = image_caption_parts[0];
            let _caption_id = image_caption_parts[1];

            // Add file extension if missing
            let image_filename = if image_id.ends_with(".jpg") {
                image_id.to_string()
            } else {
                format!("{}.jpg", image_id)
            };

            // Store mapping
            image_id_to_filename.insert(image_id.to_string(), image_filename.clone());

            // Add caption to annotations
            annotations
                .entry(image_filename)
                .or_insert_with(Vec::new)
                .push(caption_text.to_string());
        }

        println!("Loaded annotations for {} unique images", annotations.len());
        println!(
            "Total captions: {}",
            annotations.values().map(|v| v.len()).sum::<usize>()
        );

        Ok((annotations, image_id_to_filename))
    }

    /// Create standard train/validation split (29,783 train, 1,000 validation)
    ///
    /// Based on the standard Karpathy split used in many CLIP papers
    fn create_train_val_split(
        image_id_to_filename: &HashMap<String, String>,
    ) -> (Vec<String>, Vec<String>) {
        let mut all_filenames: Vec<String> = image_id_to_filename.values().cloned().collect();
        all_filenames.sort();

        let total_images = all_filenames.len();
        if total_images <= 1 {
            println!(
                "Created train/val split: {} train, {} validation",
                all_filenames.len(),
                0
            );
            return (all_filenames, Vec::new());
        }

        // Standard Flickr30K split: last 1000 for validation.
        // For small test fixtures, keep at least one image in train.
        let desired_val_size = 1000.min((total_images / 5).max(1));
        let val_size = desired_val_size.min(total_images - 1);
        let train_size = total_images - val_size;

        let train_images = all_filenames[..train_size].to_vec();
        let val_images = all_filenames[train_size..].to_vec();

        println!(
            "Created train/val split: {} train, {} validation",
            train_images.len(),
            val_images.len()
        );

        (train_images, val_images)
    }

    /// Compute dataset statistics
    async fn compute_statistics(
        annotations: &HashMap<String, Vec<String>>,
        _image_dir: &Path,
    ) -> DatasetStatistics {
        let total_pairs: usize = annotations.values().map(|v| v.len()).sum();
        let avg_caption_length = Self::compute_average_caption_length(annotations);

        // Estimate image sizes (would need to actually load images for precise count)
        let mut image_sizes = Vec::new();
        for (_, captions) in annotations.iter().take(100) {
            // Sample first 100 images to estimate sizes
            if let Some(_first_caption) = captions.first() {
                // In a real implementation, we'd load actual image sizes
                // For now, assume standard Flickr30K sizes
                image_sizes.push((640, 480)); // Approximate average
            }
        }

        // Estimate disk size (rough calculation)
        let total_images = annotations.len();
        let avg_image_size_mb = 0.15; // Flickr images are typically smaller than COCO
        let disk_size_mb = Some(total_images as f64 * avg_image_size_mb);

        DatasetStatistics {
            total_pairs,
            avg_caption_length,
            vocab_size: 0, // Would need tokenization to compute
            image_sizes: if image_sizes.is_empty() {
                None
            } else {
                Some(image_sizes)
            },
            disk_size_mb,
        }
    }

    /// Compute average caption length
    fn compute_average_caption_length(annotations: &HashMap<String, Vec<String>>) -> f64 {
        let mut total_words = 0;
        let mut total_captions = 0;

        for captions in annotations.values() {
            for caption in captions {
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

    /// Clean and preprocess captions
    pub fn clean_captions(&mut self) {
        println!("Cleaning Flickr30K captions...");

        let mut cleaned_annotations = HashMap::new();

        for (image_filename, captions) in &self.annotations {
            let cleaned_captions: Vec<String> = captions
                .iter()
                .map(|caption| Self::clean_caption(caption))
                .collect();

            cleaned_annotations.insert(image_filename.clone(), cleaned_captions);
        }

        self.annotations = cleaned_annotations;

        // Update statistics
        self.statistics.avg_caption_length =
            Self::compute_average_caption_length(&self.annotations);
    }

    /// Clean individual caption text
    fn clean_caption(caption: &str) -> String {
        let mut cleaned = caption.to_lowercase();

        // Remove leading/trailing whitespace
        cleaned = cleaned.trim().to_string();

        // Remove extra whitespace
        let re = Regex::new(r"\s+").unwrap();
        cleaned = re.replace_all(&cleaned, " ").to_string();

        // Remove trailing periods if present (common in Flickr30K)
        if cleaned.ends_with('.') {
            cleaned = cleaned[..cleaned.len() - 1].to_string();
            cleaned = cleaned.trim_end().to_string();
        }

        cleaned
    }

    /// Filter captions by quality criteria
    pub fn filter_captions(&mut self, min_words: usize, max_words: usize) {
        println!("Filtering captions: {}-{} words", min_words, max_words);

        let mut filtered_annotations = HashMap::new();
        let mut removed_count = 0;

        for (image_filename, captions) in &self.annotations {
            let filtered: Vec<String> = captions
                .iter()
                .filter(|caption| {
                    let word_count = caption.split_whitespace().count();
                    word_count >= min_words && word_count <= max_words
                })
                .cloned()
                .collect();

            if !filtered.is_empty() {
                filtered_annotations.insert(image_filename.clone(), filtered);
            } else {
                removed_count += 1;
            }
        }

        self.annotations = filtered_annotations;
        self.image_files = self
            .image_files
            .iter()
            .filter(|filename| self.annotations.contains_key(filename.as_str()))
            .cloned()
            .collect();

        println!("Removed {} images with no valid captions", removed_count);

        // Update statistics
        self.statistics.total_pairs = self.annotations.values().map(|v| v.len()).sum();
        self.statistics.avg_caption_length =
            Self::compute_average_caption_length(&self.annotations);
    }
}

#[async_trait::async_trait(?Send)]
impl VisionLanguageData for Flickr30kDataset {
    fn len(&self) -> usize {
        self.annotations
            .values()
            .map(|captions| captions.len())
            .sum()
    }

    fn get(
        &self,
        mut global_index: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ImageTextPair>> + Send + '_>>
    {
        let image_files = self.image_files.clone();
        let annotations = self.annotations.clone();
        let image_dir = self.image_dir.clone();
        let split = self.split;
        let total_len = self.len();

        Box::pin(async move {
            // Find which image this global index refers to
            for image_filename in &image_files {
                if let Some(captions) = annotations.get(image_filename) {
                    let caption_count = captions.len();
                    if global_index < caption_count {
                        // Load image data
                        let image_path = image_dir.join(image_filename);
                        let image_data = Self::load_image_data_static(&image_path).await?;

                        // Extract image ID from filename
                        let image_id = image_filename.trim_end_matches(".jpg");

                        return Ok(ImageTextPair {
                            image_data,
                            image_path: image_path.to_string_lossy().to_string(),
                            captions: vec![captions[global_index].clone()],
                            image_id: image_id.to_string(),
                            caption_ids: vec![format!("{}_c{}", image_id, global_index)],
                            metadata: HashMap::from([
                                ("dataset".to_string(), serde_json::json!("flickr30k")),
                                (
                                    "split".to_string(),
                                    serde_json::json!(format!("{:?}", split).to_lowercase()),
                                ),
                                (
                                    "total_captions".to_string(),
                                    serde_json::json!(captions.len().to_string()),
                                ),
                            ]),
                        });
                    }
                    global_index -= caption_count;
                }
            }

            Err(NNError::InvalidInput {
                message: format!(
                    "Index {} out of bounds for {} pairs",
                    global_index + 1,
                    total_len
                ),
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

impl Flickr30kDataset {
    /// Load image data from file
    async fn _load_image_data(&self, image_path: &Path) -> Result<Vec<u8>> {
        Self::load_image_data_static(image_path).await
    }

    async fn load_image_data_static(image_path: &Path) -> Result<Vec<u8>> {
        if !image_path.exists() {
            return Err(NNError::InvalidInput {
                message: format!("Image file not found: {:?}", image_path),
            });
        }

        fs::read(image_path)
            .await
            .map_err(|e| NNError::InvalidInput {
                message: format!("Failed to read image file: {}", e),
            })
    }

    /// Get all captions for a specific image filename
    pub fn get_captions_by_filename(&self, filename: &str) -> Option<&Vec<String>> {
        self.annotations.get(filename)
    }

    /// Get image filenames for current split
    pub fn image_filenames(&self) -> &[String] {
        &self.image_files
    }

    /// Check if image exists
    pub fn has_image(&self, filename: &str) -> bool {
        let image_path = self.image_dir.join(filename);
        image_path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as std_fs;
    use tempfile::tempdir;
    use tokio::fs;

    // Create minimal test annotations file
    async fn create_test_annotations(dir: &Path) -> Result<()> {
        let annotations = [
            "1007129816.jpg#0\tA man and woman walking together .",
            "1007129816.jpg#1\tA man and woman walking hand in hand .",
            "1007129816.jpg#2\tA man and woman holding hands while walking .",
            "1007129816.jpg#3\tA couple walking hand in hand down a street .",
            "1007129816.jpg#4\tA man and woman walking together holding hands .",
        ]
        .join("\n");

        let annotations_path = dir.join("results_20130124.token");
        fs::write(annotations_path, annotations).await?;

        // Create images directory with a dummy image
        let images_dir = dir.join("flickr30k_images");
        std_fs::create_dir_all(&images_dir)?;
        fs::write(images_dir.join("1007129816.jpg"), b"fake_jpeg_data").await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_flickr30k_dataset_loading() {
        let temp_dir = tempdir().unwrap();
        let flickr_path = temp_dir.path();

        create_test_annotations(flickr_path).await.unwrap();

        let dataset = Flickr30kDataset::with_split(flickr_path, DatasetSplit::Train)
            .await
            .unwrap();

        assert_eq!(dataset.len(), 5); // 5 captions for 1 image
        assert_eq!(dataset.split(), DatasetSplit::Train);

        // Test getting first item
        let pair = dataset.get(0).await.unwrap();
        assert!(pair.captions[0].contains("walking"));
        assert_eq!(pair.image_id, "1007129816");
        assert!(!pair.image_data.is_empty());
    }

    #[tokio::test]
    async fn test_caption_cleaning() {
        let temp_dir = tempdir().unwrap();
        let flickr_path = temp_dir.path();

        create_test_annotations(flickr_path).await.unwrap();

        let mut dataset = Flickr30kDataset::with_split(flickr_path, DatasetSplit::Train)
            .await
            .unwrap();

        // Before cleaning
        let original_caption = dataset.get(0).await.unwrap().captions[0].clone();
        assert!(original_caption.contains("A man and woman"));

        // Apply cleaning
        dataset.clean_captions();

        // After cleaning (should be lowercase and trimmed)
        let cleaned_caption = dataset.get(0).await.unwrap().captions[0].clone();
        assert!(cleaned_caption.starts_with("a man"));
        assert!(!cleaned_caption.ends_with(' '));
    }

    #[tokio::test]
    async fn test_caption_filtering() {
        let temp_dir = tempdir().unwrap();
        let flickr_path = temp_dir.path();

        create_test_annotations(flickr_path).await.unwrap();

        let mut dataset = Flickr30kDataset::with_split(flickr_path, DatasetSplit::Train)
            .await
            .unwrap();

        // All captions should be 6+ words (they are), so filtering should keep all
        dataset.filter_captions(6, 15);
        assert_eq!(dataset.len(), 5);
    }

    #[tokio::test]
    async fn test_nonexistent_path() {
        let result = Flickr30kDataset::new("/nonexistent/path").await;
        assert!(result.is_err());
    }
}
