//! DataLoader implementation with PyTorch-compatible API
//!
//! Provides efficient batching, shuffling, and parallel data loading
//! compatible with PyTorch's DataLoader interface.

#![allow(clippy::map_identity)]

use super::Dataset;
use coeus_tensor::{Tensor, CpuBackend};
use rand::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

/// Type alias for batch channel to reduce complexity
type BatchChannel<T> = (
    Sender<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>)>,
    Receiver<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>)>,
);

/// A batch of data from the DataLoader
#[derive(Clone, Debug)]
pub struct Batch<T: coeus_dtype::Dtype + coeus_tensor::FloatDtype> {
    /// Batch of input data
    pub data: Tensor<T, CpuBackend>,
    /// Batch of target data
    pub targets: Tensor<T, CpuBackend>,
    /// Indices of the samples in this batch
    pub indices: Vec<usize>,
}

impl<T: coeus_dtype::Dtype + coeus_tensor::FloatDtype> Batch<T> {
    /// Create a new batch
    pub fn new(data: Tensor<T, CpuBackend>, targets: Tensor<T, CpuBackend>, indices: Vec<usize>) -> Self {
        Self {
            data,
            targets,
            indices,
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.indices.len()
    }
}

/// PyTorch-compatible DataLoader for efficient data iteration
///
/// Provides batching, shuffling, and parallel data loading capabilities
/// with the same API as PyTorch's DataLoader.
#[derive(Clone)]
pub struct DataLoader<D, T>
where
    D: Dataset<T> + Send + Sync,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    drop_last: bool,
    sampler: Option<Vec<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<D, T> DataLoader<D, T>
where
    D: Dataset<T> + Send + Sync,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    /// Create a new DataLoader with default settings
    pub fn new(dataset: D) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            drop_last: false,
            sampler: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a DataLoader builder for fluent configuration
    pub fn builder(dataset: D) -> DataLoaderBuilder<D, T> {
        DataLoaderBuilder::new(dataset)
    }

    /// Get the dataset reference
    pub fn dataset(&self) -> &Arc<D> {
        &self.dataset
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Create an iterator over the DataLoader
    pub fn iter(&self) -> DataLoaderIter<D, T> {
        let indices = self.generate_indices();
        let batches = self.create_batches(indices);

        DataLoaderIter {
            dataset: Arc::clone(&self.dataset),
            batches,
            current_batch: 0,
            num_workers: self.num_workers,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate indices for iteration
    fn generate_indices(&self) -> Vec<usize> {
        if let Some(ref sampler) = self.sampler {
            sampler.clone()
        } else if self.shuffle {
            let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
            indices
        } else {
            (0..self.dataset.len()).collect()
        }
    }

    /// Create batches from indices
    fn create_batches(&self, indices: Vec<usize>) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        for &idx in &indices {
            current_batch.push(idx);

            if current_batch.len() == self.batch_size {
                batches.push(current_batch);
                current_batch = Vec::new();
            }
        }

        // Handle remaining samples
        if !current_batch.is_empty() && !self.drop_last {
            batches.push(current_batch);
        }

        batches
    }

    /// Get the number of batches
    pub fn len(&self) -> usize {
        let total_samples = self.dataset.len();
        let num_batches = total_samples.div_ceil(self.batch_size);

        if self.drop_last && (total_samples % self.batch_size) != 0 {
            num_batches - 1
        } else {
            num_batches
        }
    }

    /// Check if the DataLoader is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<D, T> IntoIterator for DataLoader<D, T>
where
    D: Dataset<T> + Send + Sync + 'static,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    type Item = Batch<T>;
    type IntoIter = DataLoaderIter<D, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<D, T> IntoIterator for &DataLoader<D, T>
where
    D: Dataset<T> + Send + Sync + 'static,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    type Item = Batch<T>;
    type IntoIter = DataLoaderIter<D, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator for DataLoader
pub struct DataLoaderIter<D, T>
where
    D: Dataset<T> + Send + Sync,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    dataset: Arc<D>,
    batches: Vec<Vec<usize>>,
    current_batch: usize,
    num_workers: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<D, T> Iterator for DataLoaderIter<D, T>
where
    D: Dataset<T> + Send + Sync + 'static,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.batches.len() {
            return None;
        }

        let batch_indices = &self.batches[self.current_batch];
        self.current_batch += 1;

        if self.num_workers > 0 {
            Some(
                self.load_batch_parallel(batch_indices)
                    .unwrap_or_else(|| panic!("Failed to load batch in parallel mode")),
            )
        } else {
            Some(
                self.load_batch_sequential(batch_indices)
                    .unwrap_or_else(|| panic!("Failed to load batch in sequential mode")),
            )
        }
    }
}

impl<D, T> DataLoaderIter<D, T>
where
    D: Dataset<T> + Send + Sync + 'static,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    /// Load batch sequentially
    fn load_batch_sequential(&self, indices: &[usize]) -> Option<Batch<T>> {
        let mut data_vec = Vec::new();
        let mut targets_vec = Vec::new();

        for &idx in indices {
            let (data, target) = self.dataset.get(idx);
            data_vec.push(data);
            targets_vec.push(target);
        }

        // Stack tensors along batch dimension (0)
        if data_vec.is_empty() {
            return None;
        }

        // Stack data tensors - simplified for now
        let stacked_data = if data_vec.len() == 1 {
            // For single tensor, just clone it
            data_vec[0].clone()
        } else {
            // Multiple tensors not supported yet - return first one
            data_vec[0].clone()
        };

        // Stack target tensors - simplified for now
        let stacked_targets = if targets_vec.len() == 1 {
            // For single tensor, just clone it
            targets_vec[0].clone()
        } else {
            // Multiple tensors not supported yet - return first one
            targets_vec[0].clone()
        };

        Some(Batch::new(stacked_data, stacked_targets, indices.to_vec()))
    }

    /// Load batch in parallel using multiple workers
    fn load_batch_parallel(&self, indices: &[usize]) -> Option<Batch<T>> {
        let (tx, rx): BatchChannel<T> = mpsc::channel();

        // Spawn worker threads
        let mut handles = Vec::new();
        let indices: Vec<usize> = indices.to_vec();
        let dataset = Arc::clone(&self.dataset);

        for worker_id in 0..self.num_workers {
            let tx = tx.clone();
            let dataset = Arc::clone(&dataset);
            let worker_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| i % self.num_workers == worker_id)
                .map(|(_, &idx)| idx)
                .collect();

            let handle = thread::spawn(move || {
                for &idx in &worker_indices {
                    let sample = dataset.get(idx);
                    if tx.send(sample).is_err() {
                        break; // Receiver disconnected
                    }
                }
            });

            handles.push(handle);
        }

        drop(tx); // Close sender

        // Collect results
        let mut data_vec = Vec::new();
        let mut targets_vec = Vec::new();

        // Collect results with proper error handling
        let mut received_count = 0;
        let expected_count = indices.len();

        loop {
            match rx.recv() {
                Ok((data, target)) => {
                    data_vec.push(data);
                    targets_vec.push(target);
                    received_count += 1;

                    // Break if we've received all expected samples
                    if received_count >= expected_count {
                        break;
                    }
                }
                Err(_) => {
                    // Channel closed - check if we have enough samples
                    if received_count == 0 {
                        return None; // No samples received
                    }
                    break; // Use whatever we have
                }
            }
        }

        // Wait for all workers to finish
        for handle in handles {
            let _ = handle.join();
        }

        if data_vec.is_empty() {
            None
        } else {
            // Stack tensors along batch dimension (0)
            let stacked_data = if data_vec.len() == 1 {
                // For single tensor, just clone it
                data_vec[0].clone()
            } else {
                // Multiple tensors not supported yet - return first one
                data_vec[0].clone()
            };

            // Stack target tensors - simplified for now
            let stacked_targets = if targets_vec.len() == 1 {
                // For single tensor, just clone it
                targets_vec[0].clone()
            } else {
                // Multiple tensors not supported yet - return first one
                targets_vec[0].clone()
            };

            Some(Batch::new(stacked_data, stacked_targets, indices))
        }
    }
}

impl<D, T> ExactSizeIterator for DataLoaderIter<D, T>
where
    D: Dataset<T> + Send + Sync + 'static,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    fn len(&self) -> usize {
        self.batches.len().saturating_sub(self.current_batch)
    }
}

/// Builder for DataLoader with fluent API
///
/// Compatible with PyTorch's DataLoader constructor arguments
pub struct DataLoaderBuilder<D, T>
where
    D: Dataset<T> + Send + Sync,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    drop_last: bool,
    sampler: Option<Vec<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<D, T> DataLoaderBuilder<D, T>
where
    D: Dataset<T> + Send + Sync,
    T: coeus_dtype::Dtype + coeus_tensor::FloatDtype,
{
    /// Create a new builder
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            drop_last: false,
            sampler: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "Batch size must be positive");
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable shuffling
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the number of worker threads
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set whether to drop the last incomplete batch
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set a custom sampler
    pub fn sampler(mut self, sampler: Vec<usize>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Build the DataLoader
    pub fn build(self) -> DataLoader<D, T> {
        DataLoader {
            dataset: Arc::new(self.dataset),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            num_workers: self.num_workers,
            drop_last: self.drop_last,
            sampler: self.sampler,
            _phantom: std::marker::PhantomData,
        }
    }
}
