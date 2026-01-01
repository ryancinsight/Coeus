//! Streaming Dataset Processing for Large-Scale Training
//!
//! This module provides streaming capabilities for processing large vision-language datasets
//! that don't fit entirely in memory. Supports distributed processing and prefetching.

use super::{ImageTextPair, VisionLanguageData};
use crate::error::{NNError, Result};
use async_stream;
use futures::stream::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, Semaphore};

/// Configuration for streaming dataset processing
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of samples to process in each streaming batch
    pub batch_size: usize,
    /// Buffer size for streaming operations
    pub buffer_size: usize,
    /// Number of worker threads for parallel processing
    pub num_workers: usize,
    /// Maximum items to process (None for all)
    pub max_items: Option<usize>,
    /// Enable shuffling of the stream
    pub shuffle: bool,
    /// Random seed for reproducible shuffling
    pub seed: u64,
    /// Timeout for streaming operations (ms)
    pub timeout_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            buffer_size: 10,
            num_workers: 4,
            max_items: None,
            shuffle: false,
            seed: 42,
            timeout_ms: 30000,
        }
    }
}

/// Streaming dataset wrapper that provides async streaming capabilities
pub struct StreamingDataset<T: VisionLanguageData + Send + Sync + 'static> {
    dataset: Arc<T>,
    config: StreamingConfig,
    current_index: std::sync::atomic::AtomicUsize,
    semaphore: Arc<Semaphore>,
}

impl<T: VisionLanguageData + Send + Sync + 'static> StreamingDataset<T> {
    /// Create a new streaming dataset wrapper
    pub fn new(dataset: T, config: StreamingConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.num_workers));

        Self {
            dataset: Arc::new(dataset),
            config,
            current_index: std::sync::atomic::AtomicUsize::new(0),
            semaphore,
        }
    }

    /// Create a stream of image-text pairs
    pub fn stream(&self) -> DatasetStream<T> {
        DatasetStream::new(
            self.dataset.clone(),
            self.config.clone(),
            self.current_index.load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    /// Create a batched stream (groups items into batches)
    pub fn batched_stream(&self) -> BatchedDatasetStream<T> {
        BatchedDatasetStream::new(self.dataset.clone(), self.config.clone())
    }

    /// Get current stream position
    pub fn position(&self) -> usize {
        self.current_index.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Reset stream position to beginning
    pub fn reset(&self) {
        self.current_index
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Stream implementation for dataset items
pub struct DatasetStream<T: VisionLanguageData + Send + Sync + 'static> {
    dataset: Arc<T>,
    config: StreamingConfig,
    current_index: usize,
    total_items: usize,
    receiver: Option<mpsc::Receiver<Result<ImageTextPair>>>,
}

impl<T: VisionLanguageData + Send + Sync + 'static> DatasetStream<T> {
    fn new(dataset: Arc<T>, config: StreamingConfig, start_index: usize) -> Self {
        let total_items = dataset.len();
        let max_items = config.max_items.unwrap_or(total_items);
        let actual_total = std::cmp::min(max_items, total_items - start_index);

        // Create channel for streaming
        let (tx, rx) = mpsc::channel(config.buffer_size);

        // Spawn streaming task
        let dataset_clone = dataset.clone();
        tokio::spawn(async move {
            let mut sent = 0;
            let mut indices: Vec<usize> = if config.shuffle {
                // Create shuffled indices
                let mut indices: Vec<usize> =
                    (start_index..std::cmp::min(start_index + max_items, total_items)).collect();
                let mut rng = rand::thread_rng();
                for i in (1..indices.len()).rev() {
                    let j = rand::Rng::gen_range(&mut rng, 0..=i);
                    indices.swap(i, j);
                }
                indices
            } else {
                (start_index..std::cmp::min(start_index + max_items, total_items)).collect()
            };

            for idx in indices {
                if sent >= actual_total {
                    break;
                }

                match dataset_clone.get(idx).await {
                    Ok(pair) => {
                        if tx.send(Ok(pair)).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
                sent += 1;
            }
        });

        Self {
            dataset,
            config,
            current_index: start_index,
            total_items: actual_total,
            receiver: Some(rx),
        }
    }
}

impl<T: VisionLanguageData + Send + Sync + 'static> Stream for DatasetStream<T> {
    type Item = Result<ImageTextPair>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(ref mut rx) = self.receiver {
            match Pin::new(rx).poll_recv(cx) {
                Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Ready(None)
        }
    }
}

/// Batched streaming for efficient processing
pub struct BatchedDatasetStream<T: VisionLanguageData + Send + Sync + 'static> {
    dataset: Arc<T>,
    config: StreamingConfig,
    receiver: Option<mpsc::Receiver<Result<Vec<ImageTextPair>>>>,
}

impl<T: VisionLanguageData + Send + Sync + 'static> BatchedDatasetStream<T> {
    fn new(dataset: Arc<T>, config: StreamingConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.buffer_size);

        let dataset_clone = dataset.clone();
        let config_clone = config.clone();

        tokio::spawn(async move {
            let total_items = dataset_clone.len();
            let max_items = config_clone.max_items.unwrap_or(total_items);
            let mut processed = 0;

            while processed < max_items && processed < total_items {
                let remaining = max_items - processed;
                let batch_size = std::cmp::min(config_clone.batch_size, remaining);

                let mut batch = Vec::with_capacity(batch_size);
                let mut batch_indices = Vec::with_capacity(batch_size);

                // Collect batch indices
                for i in 0..batch_size {
                    let idx = processed + i;
                    if idx >= total_items {
                        break;
                    }
                    batch_indices.push(idx);
                }

                if batch_indices.is_empty() {
                    break;
                }

                // Load batch in parallel
                match dataset_clone.get_batch(&batch_indices).await {
                    Ok(items) => {
                        for item in items {
                            batch.push(item);
                        }

                        if tx.send(Ok(batch)).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }

                processed += batch_indices.len();
            }
        });

        Self {
            dataset,
            config,
            receiver: Some(rx),
        }
    }
}

impl<T: VisionLanguageData + Send + Sync + 'static> Stream for BatchedDatasetStream<T> {
    type Item = Result<Vec<ImageTextPair>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(ref mut rx) = self.receiver {
            match Pin::new(rx).poll_recv(cx) {
                Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Ready(None)
        }
    }
}

/// Distributed streaming for multi-GPU/multi-node training
pub mod distributed {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Configuration for distributed streaming
    #[derive(Debug, Clone)]
    pub struct DistributedConfig {
        /// Global streaming config
        pub stream_config: StreamingConfig,
        /// Total number of workers (GPUs/nodes)
        pub world_size: usize,
        /// Rank of current worker (0 to world_size-1)
        pub rank: usize,
        /// Base seed for distributed shuffling
        pub base_seed: u64,
    }

    /// Distributed streaming dataset
    pub struct DistributedStreamingDataset<T: VisionLanguageData + Send + Sync + 'static> {
        dataset: Arc<T>,
        config: DistributedConfig,
        epoch_counter: AtomicUsize,
    }

    impl<T: VisionLanguageData + Send + Sync + 'static> DistributedStreamingDataset<T> {
        pub fn new(dataset: T, config: DistributedConfig) -> Self {
            Self {
                dataset: Arc::new(dataset),
                config,
                epoch_counter: AtomicUsize::new(0),
            }
        }

        /// Create a stream for this worker's partition of the data
        pub fn stream(&self) -> DatasetStream<T> {
            let epoch = self.epoch_counter.fetch_add(1, Ordering::SeqCst);

            // Create worker-specific config
            let mut worker_config = self.config.stream_config.clone();
            worker_config.shuffle = true;
            worker_config.seed = self.config.base_seed + epoch as u64;

            // Partition the dataset
            let total_items = self.dataset.len();
            let items_per_worker =
                (total_items + self.config.world_size - 1) / self.config.world_size;

            let start_idx = self.config.rank * items_per_worker;
            let max_items = std::cmp::min(items_per_worker, total_items - start_idx);

            worker_config.max_items = Some(max_items);

            DatasetStream::new(self.dataset.clone(), worker_config, start_idx)
        }

        /// Get worker statistics
        pub fn worker_stats(&self) -> HashMap<String, String> {
            let mut stats = HashMap::new();
            stats.insert("world_size".to_string(), self.config.world_size.to_string());
            stats.insert("rank".to_string(), self.config.rank.to_string());
            stats.insert(
                "epoch".to_string(),
                self.epoch_counter.load(Ordering::Relaxed).to_string(),
            );

            let total_items = self.dataset.len();
            let items_per_worker =
                (total_items + self.config.world_size - 1) / self.config.world_size;
            stats.insert("items_per_worker".to_string(), items_per_worker.to_string());

            stats
        }
    }
}

/// Streaming utilities and helpers
pub mod utils {
    use super::*;
    use futures::stream::BoxStream;
    use futures::StreamExt;

    /// Convert any stream to a boxed stream
    pub fn boxed_stream<T: VisionLanguageData + Send + Sync + 'static>(
        stream: DatasetStream<T>,
    ) -> BoxStream<'static, Result<ImageTextPair>> {
        stream.boxed()
    }

    /// Create a throttled stream to control processing rate
    pub fn throttled_stream<T: VisionLanguageData + Send + Sync + 'static>(
        dataset: Arc<T>,
        config: StreamingConfig,
        items_per_second: usize,
    ) -> impl Stream<Item = Result<ImageTextPair>> {
        use tokio::time::{self, Duration};

        let mut stream = DatasetStream::new(dataset, config, 0);
        let mut last_yield = time::Instant::now();

        async_stream::stream! {
            let interval = Duration::from_secs(1) / items_per_second as u32;

            while let Some(item) = stream.next().await {
                let elapsed = last_yield.elapsed();
                if elapsed < interval {
                    time::sleep(interval - elapsed).await;
                }

                yield item;
                last_yield = time::Instant::now();
            }
        }
    }

    /// Collect stream results into a vector with error handling
    pub async fn collect_stream<T: VisionLanguageData + Send + Sync + 'static>(
        stream: DatasetStream<T>,
        max_items: Option<usize>,
    ) -> Result<Vec<ImageTextPair>> {
        let mut results = Vec::new();
        futures::pin_mut!(stream);

        while let Some(item) = stream.next().await {
            match item {
                Ok(pair) => {
                    results.push(pair);
                    if max_items.is_some_and(|max| results.len() >= max) {
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }

    /// Stream processing statistics
    #[derive(Debug, Clone)]
    pub struct StreamStats {
        pub items_processed: usize,
        pub errors_encountered: usize,
        pub processing_time_ms: u64,
        pub throughput_items_per_sec: f64,
    }

    /// Monitor stream processing and collect statistics
    pub fn monitored_stream<T: VisionLanguageData + Send + Sync + 'static>(
        dataset: Arc<T>,
        config: StreamingConfig,
    ) -> impl Stream<Item = (Result<ImageTextPair>, StreamStats)> {
        use futures::stream;
        use std::sync::Mutex;
        use std::time::Instant;

        let stats = Arc::new(Mutex::new(StreamStats {
            items_processed: 0,
            errors_encountered: 0,
            processing_time_ms: 0,
            throughput_items_per_sec: 0.0,
        }));

        let start_time = Instant::now();
        let stream_instance = DatasetStream::new(dataset, config, 0);

        stream::unfold(
            (stream_instance, stats, start_time),
            |(mut stream, stats, start_time)| async move {
                match stream.next().await {
                    Some(item) => {
                        let mut current_stats = stats.lock().unwrap();
                        let elapsed = start_time.elapsed().as_millis() as u64;

                        match &item {
                            Ok(_) => current_stats.items_processed += 1,
                            Err(_) => current_stats.errors_encountered += 1,
                        }

                        current_stats.processing_time_ms = elapsed;
                        current_stats.throughput_items_per_sec = if elapsed > 0 {
                            current_stats.items_processed as f64 * 1000.0 / elapsed as f64
                        } else {
                            0.0
                        };

                        Some((
                            (item, current_stats.clone()),
                            (stream, stats.clone(), start_time),
                        ))
                    }
                    None => None,
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock dataset for testing
    struct MockDataset {
        pairs: Vec<crate::datasets::ImageTextPair>,
    }

    #[async_trait::async_trait(?Send)]
    impl crate::datasets::VisionLanguageData for MockDataset {
        fn len(&self) -> usize {
            self.pairs.len()
        }

        fn get(
            &self,
            index: usize,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = crate::error::Result<crate::datasets::ImageTextPair>,
                    > + Send
                    + '_,
            >,
        > {
            let pair = self.pairs[index].clone();
            Box::pin(async move { Ok(pair) })
        }

        fn split(&self) -> crate::datasets::DatasetSplit {
            crate::datasets::DatasetSplit::Train
        }

        fn statistics(&self) -> crate::datasets::DatasetStatistics {
            crate::datasets::DatasetStatistics {
                total_pairs: self.pairs.len(),
                avg_caption_length: 10.0, // Mock value
                vocab_size: 1000,         // Mock value
                image_sizes: Some(vec![]),
                disk_size_mb: Some(1.0), // Mock value
            }
        }
    }

    fn create_test_dataset() -> MockDataset {
        let pairs = vec![
            ImageTextPair {
                image_data: vec![1u8; 100],
                image_path: "test1.jpg".to_string(),
                captions: vec!["test caption 1".to_string()],
                image_id: "1".to_string(),
                caption_ids: vec!["1_0".to_string()],
                metadata: HashMap::new(),
            },
            ImageTextPair {
                image_data: vec![2u8; 100],
                image_path: "test2.jpg".to_string(),
                captions: vec!["test caption 2".to_string()],
                image_id: "2".to_string(),
                caption_ids: vec!["2_0".to_string()],
                metadata: HashMap::new(),
            },
        ];

        MockDataset { pairs }
    }

    #[tokio::test]
    async fn test_streaming_dataset_creation() {
        let dataset = create_test_dataset();
        let streaming_config = StreamingConfig::default();
        let streaming = StreamingDataset::new(dataset, streaming_config);

        assert_eq!(streaming.position(), 0);
    }

    #[tokio::test]
    async fn test_basic_streaming() {
        let dataset = create_test_dataset();
        let streaming_config = StreamingConfig::default();
        let streaming = StreamingDataset::new(dataset, streaming_config);
        let mut stream = streaming.stream();

        let mut count = 0;
        while let Some(item) = stream.next().await {
            assert!(item.is_ok());
            count += 1;
        }

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_batched_streaming() {
        let dataset = create_test_dataset();
        let streaming_config = StreamingConfig {
            batch_size: 1,
            ..Default::default()
        };
        let streaming = StreamingDataset::new(dataset, streaming_config);
        let mut stream = streaming.batched_stream();

        let mut total_items = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.unwrap();
            assert_eq!(batch.len(), 1);
            total_items += batch.len();
        }

        assert_eq!(total_items, 2);
    }

    #[tokio::test]
    async fn test_distributed_streaming() {
        let dataset = create_test_dataset();
        let distributed_config = distributed::DistributedConfig {
            stream_config: StreamingConfig::default(),
            world_size: 2,
            rank: 0,
            base_seed: 42,
        };

        let distributed =
            distributed::DistributedStreamingDataset::new(dataset, distributed_config);
        let mut stream = distributed.stream();

        let mut count = 0;
        while let Some(item) = stream.next().await {
            assert!(item.is_ok());
            count += 1;
        }

        // With 2 total items and 2 workers, worker 0 should get 1 item
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_stream_collection() {
        let dataset = create_test_dataset();
        let streaming_config = StreamingConfig::default();
        let streaming = StreamingDataset::new(dataset, streaming_config.clone());
        let stream = streaming.stream();

        let results = utils::collect_stream(stream, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_monitored_stream() {
        let dataset = create_test_dataset();
        let streaming_config = StreamingConfig::default();
        let streaming = StreamingDataset::new(dataset, streaming_config);
        let monitored = utils::monitored_stream(streaming.dataset, StreamingConfig::default());
        futures::pin_mut!(monitored);

        let mut count = 0;
        while let Some((item, stats)) = monitored.next().await {
            assert!(item.is_ok());
            assert_eq!(stats.items_processed, count + 1);
            count += 1;
        }

        assert_eq!(count, 2);
    }
}
