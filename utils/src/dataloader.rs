//! DataLoader for batched data iteration
//!
//! The DataLoader provides iterator-based access to datasets with automatic
//! batching, shuffling, and optional multi-threading support.

use std::marker::PhantomData;

use crate::dataset::Dataset;
use crate::error::{DataError, Result};
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};

/// Iterator-based DataLoader for batched dataset access
///
/// Provides PyTorch-compatible data loading with automatic batching,
/// shuffling, and efficient memory usage.
pub struct DataLoader<D, T> {
    dataset: D,
    sampler: Box<dyn Sampler>,
    batch_size: usize,
    _phantom: PhantomData<T>,
}

impl<D, T> DataLoader<D, T>
where
    D: Dataset<T>,
{
    /// Creates a new DataLoader with the given dataset and default settings
    ///
    /// # Arguments
    /// * `dataset` - The dataset to load from
    ///
    /// # Returns
    /// A DataLoader with batch_size=1, no shuffling, or an error if dataset is empty
    pub fn new(dataset: D) -> Result<Self> {
        Self::builder(dataset).build()
    }

    /// Creates a DataLoader builder for configuring options
    pub fn builder(dataset: D) -> DataLoaderBuilder<D, T> {
        DataLoaderBuilder::new(dataset)
    }

    /// Returns the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the dataset length
    pub fn len(&self) -> usize {
        self.sampler.len() / self.batch_size
    }

    /// Returns true if the dataloader is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Builder pattern for configuring DataLoader options
pub struct DataLoaderBuilder<D, T> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    _phantom: PhantomData<T>,
}

impl<D, T> DataLoaderBuilder<D, T>
where
    D: Dataset<T>,
{
    fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            _phantom: PhantomData,
        }
    }

    /// Sets the batch size for data loading
    ///
    /// # Arguments
    /// * `batch_size` - Number of samples per batch (must be > 0)
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enables or disables shuffling
    ///
    /// # Arguments
    /// * `shuffle` - If true, randomly shuffle data each epoch
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Builds the DataLoader with the configured options
    ///
    /// # Returns
    /// A configured DataLoader, or an error if configuration is invalid
    pub fn build(self) -> Result<DataLoader<D, T>> {
        if self.batch_size == 0 {
            return Err(DataError::invalid_batch_size(0));
        }

        if self.dataset.is_empty() {
            return Err(DataError::EmptyDataset);
        }

        let sampler: Box<dyn Sampler> = if self.shuffle {
            Box::new(RandomSampler::without_replacement(self.dataset.len()))
        } else {
            Box::new(SequentialSampler::new(self.dataset.len()))
        };

        Ok(DataLoader {
            dataset: self.dataset,
            sampler,
            batch_size: self.batch_size,
            _phantom: PhantomData,
        })
    }
}

impl<D, T> Iterator for DataLoader<D, T>
where
    D: Dataset<T>,
    T: Clone, // Clone required for batching; consider Cow for large data
{
    type Item = Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.sampler.next() {
                Some(index) => match self.dataset.get(index) {
                    Ok(sample) => batch.push(sample),
                    Err(e) => return Some(Err(e)),
                },
                None => break, // No more samples available
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(Ok(batch))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.sampler.len().saturating_sub(0); // Future: track current position
        let batches = remaining / self.batch_size;
        (batches, Some(batches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    // Simple test dataset
    struct TestDataset {
        data: Vec<usize>,
    }

    impl TestDataset {
        fn new(len: usize) -> Self {
            Self {
                data: (0..len).collect(),
            }
        }
    }

    impl Dataset<usize> for TestDataset {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn get(&self, index: usize) -> Result<usize> {
            self.data
                .get(index)
                .cloned()
                .ok_or_else(|| DataError::index_out_of_bounds(index, self.len()))
        }
    }

    #[test]
    fn test_dataloader_basic() {
        let dataset = TestDataset::new(6);
        let dataloader = DataLoader::builder(dataset).batch_size(2).build().unwrap();

        assert_eq!(dataloader.batch_size(), 2);
        assert_eq!(dataloader.len(), 3); // 6 samples / 2 batch_size = 3 batches

        let mut batches = Vec::new();
        for batch_result in dataloader {
            let batch = batch_result.unwrap();
            batches.push(batch);
        }

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0, 1]);
        assert_eq!(batches[1], vec![2, 3]);
        assert_eq!(batches[2], vec![4, 5]);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset = TestDataset::new(6);
        let dataloader = DataLoader::builder(dataset)
            .batch_size(2)
            .shuffle(true)
            .build()
            .unwrap();

        let mut all_samples = Vec::new();
        for batch_result in dataloader {
            let batch = batch_result.unwrap();
            all_samples.extend(batch);
        }

        // All samples should be present but in potentially different order
        all_samples.sort();
        assert_eq!(all_samples, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_dataloader_incomplete_batch() {
        let dataset = TestDataset::new(5); // Not divisible by batch_size
        let dataloader = DataLoader::builder(dataset).batch_size(2).build().unwrap();

        let mut batches = Vec::new();
        for batch_result in dataloader {
            let batch = batch_result.unwrap();
            batches.push(batch);
        }

        assert_eq!(batches.len(), 3); // 2 full batches + 1 partial
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[2].len(), 1); // Partial batch
        assert_eq!(batches[2], vec![4]);
    }

    #[test]
    fn test_dataloader_empty_dataset() {
        let dataset = TestDataset::new(0);
        let result = DataLoader::builder(dataset).build();
        assert!(matches!(result, Err(DataError::EmptyDataset)));
    }

    #[test]
    fn test_dataloader_invalid_batch_size() {
        let dataset = TestDataset::new(5);
        let result = DataLoader::builder(dataset).batch_size(0).build();
        assert!(matches!(
            result,
            Err(DataError::InvalidBatchSize { batch_size: 0 })
        ));
    }

    #[test]
    fn test_dataloader_single_sample_batches() {
        let dataset = TestDataset::new(3);
        let dataloader = DataLoader::new(dataset).unwrap();

        assert_eq!(dataloader.batch_size(), 1);

        let mut samples = Vec::new();
        for batch_result in dataloader {
            let batch = batch_result.unwrap();
            assert_eq!(batch.len(), 1);
            samples.push(batch[0]);
        }

        assert_eq!(samples, vec![0, 1, 2]);
    }
}
