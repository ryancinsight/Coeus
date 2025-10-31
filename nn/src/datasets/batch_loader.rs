//! Memory-Efficient Batch Loader for CLIP Training
//!
//! This module provides high-performance batch loading for vision-language tasks,
//! specifically optimized for CLIP training with memory constraints (<8GB RAM).
//! Supports parallel loading, prefetching, and automatic memory management.

use crate::error::{NNError, Result};
use super::{VisionLanguageData, ImageTextPair};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use futures::stream::{self, StreamExt};

/// Configuration for batch loading
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Batch size (number of image-text pairs per batch)
    pub batch_size: usize,
    /// Target image size for preprocessing (height, width)
    pub image_size: (usize, usize),
    /// Maximum sequence length for text tokenization
    pub max_seq_length: usize,
    /// Number of worker threads for parallel loading
    pub num_workers: usize,
    /// Prefetch buffer size (number of batches to prefetch)
    pub prefetch_size: usize,
    /// Memory limit in MB (aims for <8GB total usage)
    pub memory_limit_mb: usize,
    /// Shuffle batches during loading
    pub shuffle: bool,
    /// Drop last incomplete batch
    pub drop_last: bool,
    /// Use pinned memory for GPU transfer
    pub pin_memory: bool,
    /// Timeout for batch loading operations (ms)
    pub timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            image_size: (224, 224),
            max_seq_length: 77, // CLIP's sequence length
            num_workers: 4,
            prefetch_size: 2,
            memory_limit_mb: 8000, // <8GB target
            shuffle: true,
            drop_last: false,
            pin_memory: false,
            timeout_ms: 5000,
        }
    }
}

/// A single batch of tokenized data ready for CLIP training
#[derive(Debug, Clone)]
pub struct DatasetBatch {
    /// Image pixel data: [batch_size, height, width, 3]
    /// Layout: NCHW or NHWC depending on preprocessing
    pub images: Vec<f32>,
    /// Tokenized text sequences: [batch_size, seq_len]
    pub text_tokens: Vec<u32>,
    /// Attention masks for text: [batch_size, seq_len]
    pub text_masks: Vec<u32>,
    /// Batch size (actual, may be less than config for last batch)
    pub batch_size: usize,
    /// Image dimensions
    pub image_dims: (usize, usize),
    /// Sequence length
    pub seq_length: usize,
    /// Image-text pair IDs in this batch
    pub pair_ids: Vec<String>,
    /// Metadata for this batch
    pub metadata: HashMap<String, String>,
}

/// Memory-efficient batch loader with prefetching
pub struct VisionLanguageBatchLoader<T: VisionLanguageData + Send + Sync + 'static> {
    /// The underlying dataset
    dataset: Arc<T>,
    /// Configuration
    config: BatchConfig,
    /// Current batch index
    current_batch: usize,
    /// Total number of batches
    total_batches: usize,
    /// Shuffled indices (if shuffling enabled)
    indices: Vec<usize>,
    /// Memory management
    memory_manager: MemoryManager,
    /// Parallel loading semaphore
    semaphore: Arc<Semaphore>,
    /// Prefetch channel
    prefetch_sender: Option<tokio::sync::mpsc::Sender<Result<DatasetBatch>>>,
    prefetch_receiver: Option<tokio::sync::mpsc::Receiver<Result<DatasetBatch>>>,
}

impl<T: VisionLanguageData + Send + Sync + 'static> VisionLanguageBatchLoader<T> {
    /// Create a new batch loader
    pub fn new(dataset: T, config: BatchConfig) -> Result<Self> {
        let dataset = Arc::new(dataset);
        let total_items = dataset.len();

        // Calculate total batches
        let total_batches = if config.drop_last {
            total_items / config.batch_size
        } else {
            (total_items + config.batch_size - 1) / config.batch_size
        };

        // Create shuffled indices if requested
        let indices = if config.shuffle {
            use std::collections::HashSet;
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..total_items).collect();
            // Fisher-Yates shuffle
            for i in (1..total_items).rev() {
                let j = rand::Rng::gen_range(&mut rng, 0..=i);
                indices.swap(i, j);
            }
            indices
        } else {
            (0..total_items).collect()
        };

        // Initialize memory manager
        let memory_manager = MemoryManager::new(config.memory_limit_mb, config.prefetch_size);

        // Semaphore for limiting concurrent operations
        let semaphore = Arc::new(Semaphore::new(config.num_workers));

        Ok(Self {
            dataset,
            config,
            current_batch: 0,
            total_batches,
            indices,
            memory_manager,
            semaphore,
            prefetch_sender: None,
            prefetch_receiver: None,
        })
    }

    /// Start prefetching batches in background
    pub async fn start_prefetch(&mut self) -> Result<()> {
        let (tx, rx) = tokio::sync::mpsc::channel(self.config.prefetch_size);

        self.prefetch_sender = Some(tx);
        self.prefetch_receiver = Some(rx);

        // Spawn prefetch task
        let dataset_clone = self.dataset.clone();
        let config_clone = self.config.clone();
        let indices_clone = self.indices.clone();
        let semaphore_clone = self.semaphore.clone();
        let tx_clone = self.prefetch_sender.as_ref().unwrap().clone();

        tokio::spawn(async move {
            let mut batch_idx = 0;
            let total_batches = indices_clone.len() / config_clone.batch_size;

            while batch_idx < total_batches {
                let permit = semaphore_clone.acquire().await.unwrap();
                let start_idx = batch_idx * config_clone.batch_size;
                let end_idx = std::cmp::min(start_idx + config_clone.batch_size, indices_clone.len());

                let batch_indices: Vec<usize> = indices_clone[start_idx..end_idx].to_vec();
                let dataset = dataset_clone.clone();

                tokio::spawn(async move {
                    let result = Self::load_single_batch(&dataset, &batch_indices, &config_clone).await;
                    let _permit = permit; // Release semaphore when done
                    let _ = tx_clone.send(result).await;
                });

                batch_idx += 1;

                // Throttle if memory usage is high
                if Self::estimate_memory_usage(batch_idx) > config_clone.memory_limit_mb as f64 * 0.8 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            }
        });

        Ok(())
    }

    /// Get next batch (with prefetching if enabled)
    pub async fn next_batch(&mut self) -> Result<Option<DatasetBatch>> {
        if self.current_batch >= self.total_batches {
            return Ok(None);
        }

        // Try prefetch first
        if let Some(rx) = &mut self.prefetch_receiver {
            match tokio::time::timeout(
                tokio::time::Duration::from_millis(self.config.timeout_ms),
                rx.recv()
            ).await {
                Ok(Some(batch_result)) => {
                    self.current_batch += 1;
                    return Ok(Some(batch_result?));
                }
                Ok(None) => return Ok(None), // Channel closed
                Err(_) => {
                    // Timeout - fall back to loading
                    println!("Prefetch timeout, loading batch synchronously");
                }
            }
        }

        // Load batch synchronously
        let start_idx = self.current_batch * self.config.batch_size;
        let end_idx = std::cmp::min(start_idx + self.config.batch_size, self.indices.len());
        let batch_indices: Vec<usize> = self.indices[start_idx..end_idx].to_vec();

        let batch = Self::load_single_batch(&self.dataset, &batch_indices, &self.config).await?;
        self.current_batch += 1;

        Ok(Some(batch))
    }

    /// Get next batch synchronously (blocking)
    pub fn next_batch_blocking(&mut self) -> Result<Option<DatasetBatch>> {
        // In a real implementation, you might want to run the async version in a runtime
        // For now, just delegate to a fake async version
        futures::executor::block_on(self.next_batch())
    }

    /// Reset to beginning of dataset
    pub async fn reset(&mut self) -> Result<()> {
        self.current_batch = 0;

        // Reshuffle if enabled
        if self.config.shuffle {
            let mut rng = rand::thread_rng();
            for i in (1..self.indices.len()).rev() {
                let j = rand::Rng::gen_range(&mut rng, 0..=i);
                self.indices.swap(i, j);
            }
        }

        // Stop prefetching and restart
        self.prefetch_sender = None;
        self.prefetch_receiver = None;

        if self.config.prefetch_size > 0 {
            self.start_prefetch().await?;
        }

        Ok(())
    }

    /// Get current progress (batches processed / total batches)
    pub fn progress(&self) -> (usize, usize) {
        (self.current_batch, self.total_batches)
    }

    /// Estimate memory usage for current configuration
    pub fn estimate_memory_usage(&self) -> f64 {
        Self::estimate_memory_usage_for_config(self.current_batch, &self.config)
    }

    /// Estimate memory usage for a given config and batch index
    fn estimate_memory_usage_for_config(current_batch: usize, config: &BatchConfig) -> f64 {
        let batch_memory_mb = Self::estimate_batch_memory_mb(config);
        let prefetch_memory = batch_memory_mb * config.prefetch_size as f64;
        let current_progress_memory = batch_memory_mb * current_batch as f64;

        // Account for overhead
        let overhead_factor = 1.5;
        (prefetch_memory + current_progress_memory) * overhead_factor / 1024.0 // Convert to GB
    }

    /// Estimate memory usage per batch in MB
    fn estimate_batch_memory_mb(config: &BatchConfig) -> f64 {
        let image_bytes = config.batch_size * config.image_size.0 * config.image_size.1 * 3 * 4; // f32 = 4 bytes
        let text_bytes = config.batch_size * config.max_seq_length * 4 * 4; // tokens + masks (2) + int overhead
        let overhead_bytes = config.batch_size * 1024; // Rough overhead per sample

        ((image_bytes + text_bytes + overhead_bytes) as f64) / (1024.0 * 1024.0)
    }

    /// Load a single batch of data
    async fn load_single_batch(
        dataset: &T,
        indices: &[usize],
        config: &BatchConfig
    ) -> Result<DatasetBatch> {
        let batch_size = indices.len();

        // Pre-allocate vectors
        let mut images = Vec::with_capacity(batch_size * config.image_size.0 * config.image_size.1 * 3);
        let mut text_tokens = Vec::with_capacity(batch_size * config.max_seq_length);
        let mut text_masks = Vec::with_capacity(batch_size * config.max_seq_length);
        let mut pair_ids = Vec::with_capacity(batch_size);

        // Load pairs concurrently with limit
        let pairs: Vec<ImageTextPair> = stream::iter(indices)
            .map(|&idx| async move {
                dataset.get(idx).await
            })
            .buffer_unordered(std::cmp::min(config.num_workers, indices.len()))
            .collect::<Vec<Result<ImageTextPair>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        // Process each pair
        for (i, pair) in pairs.into_iter().enumerate() {
            // Process image (simplified - real implementation would do proper preprocessing)
            let processed_image = Self::preprocess_image(&pair.image_data, config)?;
            images.extend(processed_image);

            // Process text (simplified - real implementation would use proper tokenization)
            let (tokens, masks) = Self::preprocess_text(&pair.captions, config)?;
            text_tokens.extend(tokens);
            text_masks.extend(masks);

            // Track pair ID
            pair_ids.push(pair.image_id);
        }

        Ok(DatasetBatch {
            images,
            text_tokens,
            text_masks,
            batch_size,
            image_dims: config.image_size,
            seq_length: config.max_seq_length,
            pair_ids,
            metadata: HashMap::from([
                ("batch_index".to_string(), indices[0].to_string()),
                ("prefetched".to_string(), "false".to_string()),
            ]),
        })
    }

    /// Preprocess image data (placeholder for real image processing)
    fn preprocess_image(image_data: &[u8], config: &BatchConfig) -> Result<Vec<f32>> {
        // In real implementation, this would:
        // - Decode JPEG/PNG
        // - Resize to config.image_size
        // - Normalize pixel values
        // - Apply augmentations
        // For now, create dummy normalized data

        let pixels = config.image_size.0 * config.image_size.1 * 3;
        if image_data.is_empty() {
            // Create dummy normalized image data
            let mut dummy = Vec::with_capacity(pixels);
            for i in 0..pixels {
                // Generate pseudo-random normalized values
                let val = ((i * 31) % 255) as f32 / 255.0;
                dummy.push(val);
            }
            Ok(dummy)
        } else {
            // Convert raw bytes to f32 (simplified)
            Ok(image_data.iter().map(|&b| b as f32 / 255.0).collect())
        }
    }

    /// Preprocess text data (placeholder for real tokenization)
    fn preprocess_text(captions: &[String], config: &BatchConfig) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut tokens = Vec::new();
        let mut masks = Vec::new();

        // Use first caption (simplified)
        if let Some(caption) = captions.first() {
            let words: Vec<&str> = caption.split_whitespace().collect();
            let seq_len = std::cmp::min(words.len(), config.max_seq_length - 2); // Reserve for BOS/EOS

            // Add BOS token
            tokens.push(49406); // CLIP BOS token
            masks.push(1);

            // Add word tokens (simplified - real implementation would use proper vocab)
            for word in words.iter().take(seq_len) {
                let token = Self::word_to_token(word, config.max_seq_length);
                tokens.push(token);
                masks.push(1);
            }

            // Add EOS token
            tokens.push(49407); // CLIP EOS token
            masks.push(1);

            // Pad to max length
            while tokens.len() < config.max_seq_length {
                tokens.push(49408); // CLIP PAD token
                masks.push(0); // Mask padding
            }
        } else {
            // Empty caption case
            tokens.extend(vec![49406, 49407]); // BOS, EOS
            masks.extend(vec![1, 1]);

            while tokens.len() < config.max_seq_length {
                tokens.push(49408); // PAD
                masks.push(0);
            }
        }

        // Truncate if too long (safety check)
        tokens.truncate(config.max_seq_length);
        masks.truncate(config.max_seq_length);

        Ok((tokens, masks))
    }

    /// Simple word to token conversion (placeholder - real implementation needs vocab)
    fn word_to_token(word: &str, max_vocab: usize) -> u32 {
        // Very simplified tokenization - hash word to token ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let hash = hasher.finish();

        // Map to reasonable token range (avoid special tokens)
        (hash % (max_vocab as u64 - 100)) as u32 + 100
    }

    /// Get loader configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Get dataset reference
    pub fn dataset(&self) -> &T {
        &self.dataset
    }
}

/// Memory management utilities
struct MemoryManager {
    memory_limit_mb: usize,
    current_usage_mb: std::sync::atomic::AtomicUsize,
    prefetch_size: usize,
}

impl MemoryManager {
    fn new(memory_limit_mb: usize, prefetch_size: usize) -> Self {
        Self {
            memory_limit_mb,
            current_usage_mb: std::sync::atomic::AtomicUsize::new(0),
            prefetch_size,
        }
    }

    fn can_allocate(&self, size_mb: usize) -> bool {
        let current = self.current_usage_mb.load(std::sync::atomic::Ordering::Relaxed);
        current + size_mb <= self.memory_limit_mb
    }

    fn allocate(&self, size_mb: usize) -> bool {
        if self.can_allocate(size_mb) {
            self.current_usage_mb.fetch_add(size_mb, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn deallocate(&self, size_mb: usize) {
        self.current_usage_mb.fetch_sub(size_mb, std::sync::atomic::Ordering::Relaxed);
    }

    fn usage_mb(&self) -> usize {
        self.current_usage_mb.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::vision_language::MockDataset;

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.image_size, (224, 224));
        assert_eq!(config.max_seq_length, 77);
    }

    #[test]
    fn test_memory_estimate() {
        let config = BatchConfig::default();
        let memory_gb = VisionLanguageBatchLoader::<MockDataset>::estimate_memory_usage_for_config(1, &config);
        assert!(memory_gb > 0.0 && memory_gb < 8.0); // Should be reasonable
    }

    #[tokio::test]
    async fn test_batch_loader_creation() {
        let mock_pairs = vec![
            ImageTextPair {
                image_data: vec![1, 2, 3, 4],
                image_path: "test.jpg".to_string(),
                captions: vec!["test caption".to_string()],
                image_id: "1".to_string(),
                caption_ids: vec!["1_0".to_string()],
                metadata: HashMap::new(),
            }
        ];

        let dataset = MockDataset { pairs: mock_pairs };
        let loader = VisionLanguageBatchLoader::new(dataset, BatchConfig::default()).unwrap();

        assert_eq!(loader.total_batches, 1);
        assert_eq!(loader.current_batch, 0);
    }

    #[tokio::test]
    async fn test_batch_loading() {
        let mock_pairs = vec![
            ImageTextPair {
                image_data: vec![1, 2, 3],
                image_path: "test1.jpg".to_string(),
                captions: vec!["hello world".to_string()],
                image_id: "1".to_string(),
                caption_ids: vec!["1_0".to_string()],
                metadata: HashMap::new(),
            },
            ImageTextPair {
                image_data: vec![4, 5, 6],
                image_path: "test2.jpg".to_string(),
                captions: vec!["goodbye universe".to_string()],
                image_id: "2".to_string(),
                caption_ids: vec!["2_0".to_string()],
                metadata: HashMap::new(),
            }
        ];

        let dataset = MockDataset { pairs: mock_pairs };
        let mut loader = VisionLanguageBatchLoader::new(dataset, BatchConfig {
            batch_size: 2,
            max_seq_length: 10,
            ..Default::default()
        }).unwrap();

        let batch = loader.next_batch().await.unwrap().unwrap();

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.text_tokens.len(), 20); // 2 batches * 10 seq_len
        assert_eq!(batch.text_masks.len(), 20);
        assert_eq!(batch.pair_ids.len(), 2);
    }
}





