//! Dataset trait and implementations
//!
//! This module defines the core Dataset trait that provides PyTorch-compatible
//! data access patterns, along with common dataset implementations.

use std::marker::PhantomData;

#[allow(unused_imports)]
use crate::error::{DataError, Result};

/// PyTorch-compatible Dataset trait
///
/// Datasets provide random access to individual samples via index-based lookup.
/// This design enables efficient shuffling, batching, and parallel data loading.
pub trait Dataset<T> {
    /// Returns the total number of samples in the dataset
    fn len(&self) -> usize;

    /// Returns true if the dataset contains no samples
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the sample at the given index
    ///
    /// # Arguments
    /// * `index` - The index of the sample to retrieve (0 <= index < len())
    ///
    /// # Returns
    /// The sample data of type T, or an error if the index is out of bounds
    fn get(&self, index: usize) -> Result<T>;

    /// Optional: Returns a human-readable name for the dataset
    fn name(&self) -> &str {
        "Dataset"
    }
}

/// Extension trait providing additional Dataset functionality
pub trait DatasetExt<T>: Dataset<T> {
    /// Returns an iterator over all samples in the dataset
    ///
    /// Note: This creates a new iterator each time it's called.
    /// For efficient iteration, use DataLoader instead.
    fn iter(&self) -> DatasetIter<'_, Self, T>
    where
        Self: Sized,
    {
        DatasetIter {
            dataset: self,
            index: 0,
            _phantom: PhantomData,
        }
    }
}

// Blanket implementation of DatasetExt for all Dataset implementations
impl<D, T> DatasetExt<T> for D where D: Dataset<T> {}

/// Iterator over a dataset
pub struct DatasetIter<'a, D, T> {
    dataset: &'a D,
    index: usize,
    _phantom: PhantomData<T>,
}

impl<'a, D, T> Iterator for DatasetIter<'a, D, T>
where
    D: Dataset<T>,
    T: 'a,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            None
        } else {
            let result = self.dataset.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, D, T> ExactSizeIterator for DatasetIter<'a, D, T>
where
    D: Dataset<T>,
    T: 'a,
{
    fn len(&self) -> usize {
        self.dataset.len().saturating_sub(self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test dataset that returns its index as data
    struct TestDataset {
        len: usize,
    }

    impl Dataset<usize> for TestDataset {
        fn len(&self) -> usize {
            self.len
        }

        fn get(&self, index: usize) -> Result<usize> {
            if index >= self.len {
                Err(DataError::index_out_of_bounds(index, self.len))
            } else {
                Ok(index)
            }
        }

        fn name(&self) -> &str {
            "TestDataset"
        }
    }

    #[test]
    fn test_dataset_basic_operations() {
        let dataset = TestDataset { len: 5 };

        assert_eq!(dataset.len(), 5);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.name(), "TestDataset");

        // Test valid indices
        assert_eq!(dataset.get(0).unwrap(), 0);
        assert_eq!(dataset.get(4).unwrap(), 4);

        // Test invalid index
        assert!(dataset.get(5).is_err());
    }

    #[test]
    fn test_dataset_iterator() {
        let dataset = TestDataset { len: 3 };
        let mut iter = dataset.iter();

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next().unwrap().unwrap(), 0);
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next().unwrap().unwrap(), 1);
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next().unwrap().unwrap(), 2);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = TestDataset { len: 0 };

        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());
        assert!(dataset.get(0).is_err());
        assert!(dataset.iter().next().is_none());
    }
}
