//! Subset dataset implementation
//!
//! Provides subset functionality compatible with PyTorch's Subset.

use super::dataset_trait::Dataset;
use coeus_tensor::{Tensor, CpuBackend};

/// Subset of a dataset
///
/// Compatible with PyTorch's `Subset`
#[derive(Clone)]
pub struct Subset<D, T>
where
    D: Dataset<T> + Clone,
    T: coeus_dtype::Dtype,
{
    dataset: D,
    indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<D, T> Subset<D, T>
where
    D: Dataset<T> + Clone,
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
    D: Dataset<T> + Clone,
    T: coeus_dtype::Dtype,
{
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> (Tensor<T, CpuBackend>, Tensor<T, CpuBackend>) {
        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}
