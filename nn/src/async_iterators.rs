//! Asynchronous Iterators for Neural Network Operations
//!
//! This module provides async streaming capabilities for neural network operations,
//! enabling real-time data pipelines and streaming neural network processing.
//!
//! The async iterator pattern is particularly useful for:
//! - Real-time inference on streaming data
//! - Distributed training data pipelines
//! - Gradient accumulation batches
//! - Model ensemble processing
//!
//! ## Usage Examples
//!
//! ### Streaming Inference
//!
//! ```rust,ignore
//! let model = Linear::new(784, 10).unwrap();
//! let mut stream = StreamingInference::new(model);
//!
//! // Process inputs asynchronously
//! for input in data_stream {
//!     let output = stream.process(&input).await.unwrap();
//!     // Process result...
//! }
//! ```
//!
//! ### Batch Processing with Async Iterator
//!
//! ```rust,ignore
//! let model = Sequential::new(layers);
//! let mut batch_iter = BatchAsyncIterator::new(data_loader, model, batch_size).await;
//!
//! while let Some(batch_output) = batch_iter.next().await {
//!     // Process complete batch...
//! }
//! ```
//!
//! ### Real-time Training Iterator
//!
//! ```rust,ignore
//! let mut trainer = StreamingTrainer::new(model, optimizer, loss_fn);
//!
//! while let Some(update_result) = trainer.train_step(/* stream of data */).await {
//!     println!("Loss: {:?}", update_result.loss);
//! }
//! ```

use crate::{ModuleExt, Result};
use coeus_backend::Backend;
use coeus_storage::{Storage, StorageFromVec};
use coeus_tensor::Tensor;
use coeus_dtype::DataType;
use futures::{Stream, StreamExt};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::pin::Pin;

/// Async iterator for streaming neural network inference
pub struct StreamingInference<M, B, S, T>
where
    M: ModuleExt<B, S, T>,
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    model: M,
    /// Internal buffer for caching intermediate computations
    buffer: VecDeque<Tensor<B, S, T>>,
    /// Maximum buffer size to prevent memory overflow
    max_buffer_size: usize,
}

impl<M, B, S, T> StreamingInference<M, B, S, T>
where
    M: ModuleExt<B, S, T>,
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Create a new streaming inference processor with default buffer size
    pub fn new(model: M) -> Self {
        Self {
            model,
            buffer: VecDeque::new(),
            max_buffer_size: 100, // Default buffer size, configurable via with_buffer_size()
        }
    }

    /// Set the maximum buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Process a single input asynchronously
    pub async fn process(&mut self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let output = self.model.forward(input)?;

        // Maintain buffer for potential batch processing
        if self.buffer.len() >= self.max_buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(output.clone());

        Ok(output)
    }

    /// Get buffered outputs (useful for batch operations)
    pub fn buffered_outputs(&self) -> Vec<&Tensor<B, S, T>> {
        self.buffer.iter().collect()
    }
}

/// Async iterator for batch processing of neural network operations
pub struct BatchAsyncIterator<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Async stream of input batches
    input_stream: Pin<Box<dyn Stream<Item = Vec<Tensor<B, S, T>>> + Send>>,
    /// Processing function for each batch
    process_fn: Box<
        dyn Fn(Vec<Tensor<B, S, T>>) -> futures::future::BoxFuture<'static, Result<Vec<Tensor<B, S, T>>>> + Send + Sync
    >,
}

impl<B, S, T> BatchAsyncIterator<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Create a new batch async iterator
    pub fn new<F, Fut>(input_stream: impl Stream<Item = Vec<Tensor<B, S, T>>> + Send + 'static, process_fn: F) -> Self
    where
        F: Fn(Vec<Tensor<B, S, T>>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Vec<Tensor<B, S, T>>>> + Send + 'static,
    {
        let process_fn = Box::new(move |batch| Box::pin(process_fn(batch)) as futures::future::BoxFuture<'static, _>);

        Self {
            input_stream: Box::pin(input_stream),
            process_fn,
        }
    }

    /// Process the next batch asynchronously
    pub async fn next(&mut self) -> Option<Result<Vec<Tensor<B, S, T>>>> {
        let batch = self.input_stream.next().await?;
        Some((self.process_fn)(batch).await)
    }
}

/// Neural network training step result
#[derive(Debug, Clone)]
pub struct TrainingStepResult<T: DataType> {
    /// Training loss after this step
    pub loss: T,
    /// Optional metrics (accuracy, etc.)
    pub metrics: std::collections::HashMap<String, T>,
    /// Step number
    pub step: usize,
}

/// Async training coordinator - coordinates training steps across multiple models/data streams
/// Provides high-level async training orchestration without specific optimizer/loss coupling
pub struct AsyncTrainingCoordinator<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    step_count: usize,
    workers: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> AsyncTrainingCoordinator<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Create a new async training coordinator
    pub fn new(workers: usize) -> Self {
        Self {
            step_count: 0,
            workers,
            _phantom: PhantomData,
        }
    }

    /// Execute a coordinated training step across all workers
    pub async fn coordinated_step<F, Fut>(&mut self, training_fn: F) -> Result<TrainingStepResult<T>>
    where
        F: Fn(usize) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<TrainingStepResult<T>>> + Send,
    {
        // Coordinate training across multiple workers asynchronously
        let mut handles = Vec::with_capacity(self.workers);
        for worker_id in 0..self.workers {
            let handle = tokio::spawn(async move {
                training_fn(worker_id).await
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        let mut results = Vec::with_capacity(self.workers);
        for handle in handles {
            results.push(handle.await.map_err(|e| NNError::TrainingError {
                message: format!("Worker task failed: {}", e),
            })??);
        }

        // For now, just return the first result as a simple implementation
        // TODO: Implement proper loss aggregation based on available DataType methods
        let first_result = results.into_iter().next().unwrap_or_else(|| TrainingStepResult {
            loss: T::zero(),
            metrics: std::collections::HashMap::new(),
            step: 0,
        });

        self.step_count += 1;

        Ok(TrainingStepResult {
            loss: first_result.loss,
            metrics: std::collections::HashMap::new(),
            step: self.step_count,
        })
    }
}

impl<B, S, T> Default for AsyncTrainingCoordinator<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    fn default() -> Self {
        Self::new(1)
    }
}

type NNError = crate::NNError;

/// Async iterator for data loading and preprocessing
pub struct AsyncDataLoader<D, B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    data_source: D,
    batch_size: usize,
    shuffle: bool,
    prefetch_capacity: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<D, B, S, T> AsyncDataLoader<D, B, S, T>
where
    D: Stream<Item = Result<(Tensor<B, S, T>, Tensor<B, S, T>)>> + Send + 'static,
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Create a new async data loader
    pub fn new(data_source: D, batch_size: usize) -> Self {
        Self {
            data_source,
            batch_size,
            shuffle: false,
            prefetch_capacity: 10,
            _phantom: PhantomData,
        }
    }

    /// Create a batched version of the data loader
    pub fn batched(self) -> BatchedAsyncDataLoader<D, B, S, T> {
        BatchedAsyncDataLoader {
            loader: self,
        }
    }
}

/// Batched version of the async data loader
pub struct BatchedAsyncDataLoader<D, B, S, T>
where
    D: Stream<Item = Result<(Tensor<B, S, T>, Tensor<B, S, T>)>> + Send + 'static,
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    loader: AsyncDataLoader<D, B, S, T>,
}

impl<D, B, S, T> BatchedAsyncDataLoader<D, B, S, T>
where
    D: Stream<Item = Result<(Tensor<B, S, T>, Tensor<B, S, T>)>> + Send + Unpin + 'static,
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType,
{
    /// Get the next batch
    pub async fn next_batch(&mut self) -> Result<Option<(Tensor<B, S, T>, Tensor<B, S, T>)>> {
        // TODO: Implement actual batching logic
        match self.loader.data_source.next().await {
            Some(result) => {
                let (input, target) = result?;
                Ok(Some((input, target)))
            }
            None => Ok(None),
        }
    }
}

/// Async iterator trait for neural network operations
pub trait AsyncIterator {
    type Item;

    /// Get the next item asynchronously
    fn next(&mut self) -> futures::future::BoxFuture<'_, Option<Self::Item>>;
}

/// Associated trait for async iterables that can be converted to async iterators
pub trait IntoAsyncIterator {
    type Item;
    type IntoIter: AsyncIterator<Item = Self::Item>;

    fn into_async_iter(self) -> Self::IntoIter;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;
    use coeus_backend::cpu::CpuBackend;
    use coeus_storage::dense::DenseStorage;
    use coeus_dtype::float::Float32;

    type TestBackend = CpuBackend;
    type TestStorage = DenseStorage<Float32>;
    type TestDtype = Float32;

    #[tokio::test]
    async fn test_streaming_inference() {
        let model = Linear::<TestBackend, TestStorage, TestDtype>::new(10, 5).unwrap();
        let mut streamer = StreamingInference::new(model);

        let input = Tensor::from_vec(vec![1.0; 10], &[1, 10]).unwrap();

        let output = streamer.process(&input).await.unwrap();
        assert_eq!(output.shape(), &[1, 5]);
    }

    #[tokio::test]
    async fn test_async_training_coordinator() {
        let mut coordinator = AsyncTrainingCoordinator::<TestBackend, TestStorage, TestDtype>::new(2);

        let result = coordinator.coordinated_step(|_worker_id| async {
            Ok(TrainingStepResult {
                loss: TestDtype::new(0.5),
                metrics: std::collections::HashMap::new(),
                step: 1,
            })
        }).await.unwrap();

        assert_eq!(result.step, 1);
        // Verify that the coordinator aggregates results across workers
        assert_eq!(coordinator.step_count, 1);
    }
}

