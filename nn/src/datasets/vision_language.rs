//! Core Vision-Language Dataset Abstractions
//!
//! This module provides the fundamental abstractions for vision-language datasets
//! and processing pipelines. It defines common interfaces and utilities used by
//! specific dataset implementations like COCO and Flickr30K.

use crate::error::{NNError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

/// Re-export common types
pub use super::{VisionLanguageData, DatasetSplit, DatasetStatistics};
use super::ImageTextPair;

/// Wrapper for vision-language datasets providing additional utilities
pub struct VisionLanguageDataset<T> {
    dataset: T,
}

impl<T: VisionLanguageData> VisionLanguageDataset<T> {
    /// Create a new wrapper around a vision-language dataset
    pub fn new(dataset: T) -> Self {
        Self { dataset }
    }

    /// Get inner dataset
    pub fn inner(&self) -> &T {
        &self.dataset
    }

    /// Get inner dataset mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.dataset
    }

    /// Get dataset statistics with formatted display
    pub fn print_statistics(&self) {
        let stats = self.dataset.statistics();
        let split = self.dataset.split();

        println!("=== Dataset Statistics ===");
        println!("Split: {:?}", split);
        println!("Total Pairs: {}", stats.total_pairs);
        println!("Average Caption Length: {:.1} words", stats.avg_caption_length);
        println!("Vocabulary Size: {}", stats.vocab_size);
        println!("Disk Size: {:.1} MB", stats.disk_size_mb.unwrap_or(0.0));

        if let Some(sizes) = &stats.image_sizes {
            if !sizes.is_empty() {
                let avg_width = sizes.iter().map(|(w, _)| *w).sum::<u32>() as f64 / sizes.len() as f64;
                let avg_height = sizes.iter().map(|(_, h)| *h).sum::<u32>() as f64 / sizes.len() as f64;
                println!("Average Image Size: {:.0}x{:.0}", avg_width, avg_height);
            }
        }
    }

    /// Create iterator over the dataset
    pub fn iter(&self) -> DatasetIterator<T> {
        DatasetIterator {
            dataset: &self.dataset,
            current_index: 0,
            len: self.dataset.len(),
        }
    }
}

/// Iterator over vision-language dataset
pub struct DatasetIterator<'a, T> {
    dataset: &'a T,
    current_index: usize,
    len: usize,
}

impl<'a, T: VisionLanguageData> Iterator for DatasetIterator<'a, T> {
    type Item = Result<ImageTextPair>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.len {
            None
        } else {
            // Note: This is synchronous iteration over async data
            // In practice, you might want to use streams/futures
            // For now, return an error indicating async iteration is needed
            self.current_index += 1;
            Some(Err(crate::error::NNError::NotImplemented { operation: "Async iteration required".to_string() }))
        }
    }
}

impl<'a, T: VisionLanguageData> ExactSizeIterator for DatasetIterator<'a, T> {
    fn len(&self) -> usize {
        self.len - self.current_index
    }
}

/// Stream-based processing for large datasets
pub mod streaming {
    use super::*;
    use futures::stream::{Stream, StreamExt};
    use tokio::sync::mpsc;

    /// Configuration for streaming dataset processing
    #[derive(Debug, Clone)]
    pub struct StreamConfig {
        /// Buffer size for prefetching
        pub buffer_size: usize,
        /// Number of worker threads for parallel processing
        pub num_workers: usize,
        /// Maximum items to process
        pub max_items: Option<usize>,
    }

    impl Default for StreamConfig {
        fn default() -> Self {
            Self {
                buffer_size: 100,
                num_workers: 4,
                max_items: None,
            }
        }
    }

    /// Create a stream from a vision-language dataset
    pub fn stream_dataset<T: VisionLanguageData + Send + Sync + 'static>(
        dataset: Arc<T>,
        config: StreamConfig,
    ) -> impl Stream<Item = Result<ImageTextPair>> {
        let (tx, rx) = mpsc::channel(config.buffer_size);

        tokio::spawn(async move {
            let max_items = config.max_items.unwrap_or(dataset.len());
            let mut tasks = vec![];

            for worker_id in 0..config.num_workers {
                let tx_clone = tx.clone();
                let dataset_clone = dataset.clone();
                let start_idx = worker_id;
                let step = config.num_workers;

                let task = tokio::spawn(async move {
                    let mut idx = start_idx;
                    while idx < max_items {
                        let pair = dataset_clone.get(idx).await;
                        if tx_clone.send(pair).await.is_err() {
                            break; // Receiver dropped
                        }
                        idx += step;
                    }
                });
                tasks.push(task);
            }

            // Wait for all workers to complete
            for task in tasks {
                let _ = task.await;
            }
        });

        ReceiverStream::new(rx)
    }

    /// Stream with transformations applied
    pub fn stream_with_transform<T, F, Fut>(
        dataset: Arc<T>,
        config: StreamConfig,
        transform: F,
    ) -> impl Stream<Item = Result<ImageTextPair>>
    where
        T: VisionLanguageData + Send + Sync + 'static,
        F: Fn(ImageTextPair) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<ImageTextPair>> + Send + Sync,
    {
        stream_dataset(dataset, config)
            .then(move |pair_result| {
                let transform_clone = transform.clone();
                async move {
                    match pair_result {
                        Ok(pair) => transform_clone(pair).await,
                        Err(e) => Err(e),
                    }
                }
            })
    }
}

/// Quality analysis and filtering utilities
pub mod quality {
    use super::*;
    use std::collections::HashSet;

    /// Quality metrics for dataset analysis
    #[derive(Debug, Clone)]
    pub struct QualityMetrics {
        /// Language diversity (unique words / total words)
        pub language_diversity: f64,
        /// Average caption length in words
        pub avg_caption_length: f64,
        /// Caption length variance
        pub caption_length_variance: f64,
        /// Duplicate captions percentage
        pub duplicate_percentage: f64,
        /// Missing image files
        pub missing_images: usize,
        /// Corrupt image files
        pub corrupt_images: usize,
        /// Short captions (< 3 words)
        pub short_captions: usize,
        /// Very long captions (> 50 words)
        pub long_captions: usize,
    }

    /// Analyze dataset quality
    pub async fn analyze_quality<T: VisionLanguageData>(
        dataset: &T,
        sample_size: Option<usize>,
    ) -> Result<QualityMetrics> {
        let sample_size = sample_size.unwrap_or(1000).min(dataset.len());

        println!("Analyzing dataset quality with sample of {} items...", sample_size);

        let mut all_words = HashSet::new();
        let mut total_words = 0;
        let mut caption_lengths = Vec::new();
        let mut all_captions = HashSet::new();
        let mut duplicate_captions = 0;

        let mut missing_images = 0;
        let mut corrupt_images = 0;
        let mut short_captions = 0;
        let mut long_captions = 0;

        // Sample analysis
        let step = dataset.len() / sample_size;
        for i in (0..dataset.len()).step_by(step).take(sample_size) {
            match dataset.get(i).await {
                Ok(pair) => {
                    // Check image data
                    if pair.image_data.is_empty() {
                        missing_images += 1;
                    }
                    // TODO: Add more sophisticated image corruption detection

                    // Analyze captions
                    for caption in &pair.captions {
                        let word_count = caption.split_whitespace().count();

                        // Track caption lengths
                        caption_lengths.push(word_count);

                        // Check quality thresholds
                        if word_count < 3 {
                            short_captions += 1;
                        }
                        if word_count > 50 {
                            long_captions += 1;
                        }

                        // Count words and duplicates
                        for word in caption.split_whitespace() {
                            all_words.insert(word.to_lowercase());
                            total_words += 1;
                        }

                        // Check for duplicates
                        if !all_captions.insert(caption.clone()) {
                            duplicate_captions += 1;
                        }
                    }
                }
                Err(_) => corrupt_images += 1,
            }
        }

        // Compute metrics
        let language_diversity = if total_words > 0 {
            all_words.len() as f64 / total_words as f64
        } else {
            0.0
        };

        let avg_caption_length = if !caption_lengths.is_empty() {
            caption_lengths.iter().sum::<usize>() as f64 / caption_lengths.len() as f64
        } else {
            0.0
        };

        let caption_length_variance = if caption_lengths.len() > 1 {
            let mean = avg_caption_length;
            caption_lengths.iter()
                .map(|&len| (len as f64 - mean).powi(2))
                .sum::<f64>() / (caption_lengths.len() - 1) as f64
        } else {
            0.0
        };

        let duplicate_percentage = if all_captions.len() > 0 {
            (duplicate_captions as f64 / (all_captions.len() + duplicate_captions) as f64) * 100.0
        } else {
            0.0
        };

        // Scale to full dataset
        let scale_factor = dataset.len() as f64 / sample_size as f64;
        missing_images = (missing_images as f64 * scale_factor) as usize;
        corrupt_images = (corrupt_images as f64 * scale_factor) as usize;

        Ok(QualityMetrics {
            language_diversity,
            avg_caption_length,
            caption_length_variance,
            duplicate_percentage,
            missing_images,
            corrupt_images,
            short_captions,
            long_captions,
        })
    }

    impl QualityMetrics {
        /// Print quality analysis report
        pub fn print_report(&self) {
            println!("=== Dataset Quality Analysis ===");
            println!("Language Diversity: {:.3} (unique/total words)", self.language_diversity);
            println!("Average Caption Length: {:.1} words", self.avg_caption_length);
            println!("Caption Length Variance: {:.1}", self.caption_length_variance);
            println!("Duplicate Captions: {:.2}%", self.duplicate_percentage);
            println!("Missing Images: {}", self.missing_images);
            println!("Corrupt Images: {}", self.corrupt_images);
            println!("Short Captions (< 3 words): {}", self.short_captions);
            println!("Long Captions (> 50 words): {}", self.long_captions);

            // Quality assessment
            let mut issues = Vec::new();
            if self.language_diversity < 0.1 {
                issues.push("Low language diversity - captions may be repetitive");
            }
            if self.avg_caption_length < 5.0 {
                issues.push("Average caption length is low - may lack detail");
            }
            if self.duplicate_percentage > 20.0 {
                issues.push("High duplicate caption percentage");
            }
            if self.missing_images > 0 {
                issues.push("Missing image files detected");
            }

            if issues.is_empty() {
                println!("✓ Dataset quality appears good");
            } else {
                println!("⚠ Quality issues detected:");
                for issue in issues {
                    println!("  - {}", issue);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mock dataset for testing
    struct MockDataset {
        pairs: Vec<ImageTextPair>,
    }

    #[async_trait::async_trait(?Send)]
    impl VisionLanguageData for MockDataset {
        fn len(&self) -> usize {
            self.pairs.len()
        }

        fn get(&self, index: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ImageTextPair>> + Send + '_>> {
            let pairs = self.pairs.clone();
            Box::pin(async move {
                pairs.get(index).cloned().ok_or_else(|| NNError::InvalidInput {
                    message: format!("Index out of bounds: {}", index),
                })
            })
        }

        fn split(&self) -> DatasetSplit {
            DatasetSplit::Train
        }

        fn statistics(&self) -> DatasetStatistics {
            DatasetStatistics {
                total_pairs: self.pairs.len(),
                avg_caption_length: 10.0,
                vocab_size: 100,
                image_sizes: Some(vec![(640, 480)]),
                disk_size_mb: Some(50.0),
            }
        }
    }

    #[test]
    fn test_dataset_wrapper() {
        let mock_pairs = vec![
            ImageTextPair {
                image_data: vec![1, 2, 3],
                image_path: "test1.jpg".to_string(),
                captions: vec!["Test caption 1".to_string()],
                image_id: "1".to_string(),
                caption_ids: vec!["1_0".to_string()],
                metadata: HashMap::new(),
            }
        ];

        let dataset = VisionLanguageDataset::new(MockDataset { pairs: mock_pairs });
        assert_eq!(dataset.inner().len(), 1);

        // Test iterator
        let mut iter = dataset.iter();
        assert_eq!(iter.len(), 1);
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    #[tokio::test]
    async fn test_quality_analysis() {
        let mock_pairs = vec![
            ImageTextPair {
                image_data: vec![1, 2, 3],
                image_path: "test.jpg".to_string(),
                captions: vec!["This is a test caption".to_string()],
                image_id: "1".to_string(),
                caption_ids: vec!["1_0".to_string()],
                metadata: HashMap::new(),
            },
            ImageTextPair {
                image_data: vec![4, 5, 6],
                image_path: "test2.jpg".to_string(),
                captions: vec!["Another test caption here".to_string()],
                image_id: "2".to_string(),
                caption_ids: vec!["2_0".to_string()],
                metadata: HashMap::new(),
            }
        ];

        let dataset = MockDataset { pairs: mock_pairs };
        let quality = quality::analyze_quality(&dataset, None).await.unwrap();

        // Should compute reasonable metrics
        assert!(quality.avg_caption_length > 3.0);
        assert!(quality.language_diversity > 0.0);
    }
}





