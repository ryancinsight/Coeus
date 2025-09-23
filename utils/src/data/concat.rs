//! ConcatDataset implementation
//!
//! Provides dataset concatenation functionality compatible with PyTorch's ConcatDataset.

use super::dataset_trait::Dataset;
use coeus_tensor::Tensor;

/// Concatenation of multiple datasets
///
/// Compatible with PyTorch's `ConcatDataset`
pub struct ConcatDataset<T: coeus_dtype::Dtype> {
    datasets: Vec<Box<dyn Dataset<T>>>,
    cumulative_lengths: Vec<usize>,
}

impl<T: coeus_dtype::Dtype> ConcatDataset<T> {
    /// Create a new concatenated dataset
    pub fn new(datasets: Vec<Box<dyn Dataset<T>>>) -> Self {
        let mut cumulative_lengths = Vec::with_capacity(datasets.len());
        let mut total_len = 0;

        for dataset in &datasets {
            total_len += dataset.len();
            cumulative_lengths.push(total_len);
        }

        Self {
            datasets,
            cumulative_lengths,
        }
    }

    /// Find which dataset and local index corresponds to a global index
    fn find_dataset_index(&self, global_index: usize) -> (usize, usize) {
        // Binary search to find the dataset
        let mut left = 0;
        let mut right = self.cumulative_lengths.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if global_index < self.cumulative_lengths[mid] {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        let dataset_index = left;
        let local_index = if dataset_index == 0 {
            global_index
        } else {
            global_index - self.cumulative_lengths[dataset_index - 1]
        };

        (dataset_index, local_index)
    }
}

impl<T: coeus_dtype::Dtype> Dataset<T> for ConcatDataset<T> {
    fn len(&self) -> usize {
        self.cumulative_lengths.last().copied().unwrap_or(0)
    }

    fn get(&self, index: usize) -> (Tensor<T>, Tensor<T>) {
        let (dataset_index, local_index) = self.find_dataset_index(index);
        self.datasets[dataset_index].get(local_index)
    }
}
