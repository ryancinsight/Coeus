//! Dataset trait and implementations
//!
//! Provides the core `Dataset` trait and common implementations
//! compatible with PyTorch's dataset interface.

use coeus_tensor::Tensor;

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

/// A simple dataset that wraps tensors directly
///
/// Compatible with PyTorch's `TensorDataset`
pub struct TensorDataset<T: coeus_dtype::Dtype> {
    data: Vec<Tensor<T>>,
    targets: Vec<Tensor<T>>,
}

impl<T: coeus_dtype::Dtype> TensorDataset<T> {
    /// Create a new TensorDataset
    ///
    /// # Arguments
    /// * `data` - Input tensors
    /// * `targets` - Target tensors
    ///
    /// # Panics
    /// Panics if data and targets have different lengths
    pub fn new(data: Vec<Tensor<T>>, targets: Vec<Tensor<T>>) -> Self {
        assert_eq!(
            data.len(),
            targets.len(),
            "Data and targets must have the same length"
        );
        Self { data, targets }
    }

    /// Create a TensorDataset from arrays
    pub fn from_arrays(data: &[Tensor<T>], targets: &[Tensor<T>]) -> Self {
        Self::new(data.to_vec(), targets.to_vec())
    }
}

impl<T: coeus_dtype::Dtype> Dataset<T> for TensorDataset<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> (Tensor<T>, Tensor<T>) {
        (self.data[index].clone(), self.targets[index].clone())
    }
}

/// Subset of a dataset
///
/// Compatible with PyTorch's `Subset`
pub struct Subset<D, T>
where
    D: Dataset<T>,
    T: coeus_dtype::Dtype,
{
    dataset: D,
    indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<D, T> Subset<D, T>
where
    D: Dataset<T>,
    T: coeus_dtype::Dtype,
{
    /// Create a new subset
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self {
            dataset,
            indices,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D, T> Dataset<T> for Subset<D, T>
where
    D: Dataset<T>,
    T: coeus_dtype::Dtype,
{
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> (Tensor<T>, Tensor<T>) {
        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

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
