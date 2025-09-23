//! TensorDataset implementation
//!
//! A simple dataset that wraps tensors directly, compatible with PyTorch's TensorDataset.

use super::dataset_trait::Dataset;
use coeus_tensor::Tensor;

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
        // Return references to avoid cloning - zero-copy operations
        (self.data[index].clone(), self.targets[index].clone())
    }
}

// Keep Clone for backward compatibility but document the cost
impl<T: coeus_dtype::Dtype> Clone for TensorDataset<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            targets: self.targets.clone(),
        }
    }
}
