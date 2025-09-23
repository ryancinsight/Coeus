//! Dataset trait and iterator implementation
//!
//! Core abstractions for dataset functionality compatible with PyTorch's Dataset interface.

use crate::Tensor;

/// Core trait for datasets, compatible with PyTorch's Dataset interface
pub trait Dataset<T: coeus_dtype::Dtype>: Send + Sync {
    /// Returns the total number of samples in the dataset
    fn len(&self) -> usize;

    /// Returns the sample at the given index
    fn get(&self, index: usize) -> (Tensor<T>, Tensor<T>);

    /// Returns true if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset
    fn iter(&self) -> DatasetIter<'_, T>
    where
        Self: Sized,
    {
        DatasetIter {
            dataset: self,
            index: 0,
        }
    }
}

/// Iterator for datasets
pub struct DatasetIter<'a, T: coeus_dtype::Dtype> {
    dataset: &'a dyn Dataset<T>,
    index: usize,
}

impl<'a, T: coeus_dtype::Dtype> Iterator for DatasetIter<'a, T> {
    type Item = (Tensor<T>, Tensor<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.len() {
            let item = self.dataset.get(self.index);
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<'a, T: coeus_dtype::Dtype> ExactSizeIterator for DatasetIter<'a, T> {
    fn len(&self) -> usize {
        self.dataset.len().saturating_sub(self.index)
    }
}
