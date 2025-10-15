//! TensorDataset implementation
//!
//! A dataset that wraps tensors for easy data loading.

use std::sync::Arc;

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_dtype::int::Int32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use crate::dataset::Dataset;
use crate::error::{DataError, Result};

/// A dataset that combines multiple tensors into a single dataset
///
/// Each sample consists of one element from each input tensor.
/// All tensors must have the same length in their first dimension.
///
/// # Example
///
/// ```rust
/// use coeus_utils::{Dataset, TensorDataset};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
/// use coeus_dtype::int::Int32;
///
/// // Create input data and targets
/// let data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)], &[4]).unwrap();
/// let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(vec![Int32::new(0), Int32::new(1), Int32::new(0), Int32::new(1)], &[4]).unwrap();
///
/// // Create dataset
/// let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();
/// assert_eq!(dataset.len(), 4);
///
/// // Get first sample
/// let sample = dataset.get(0).unwrap();
/// // sample contains tensors for data and targets
/// ```
pub struct TensorDataset {
    /// Input tensors (features)
    inputs: Vec<Arc<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>>,
    /// Target tensors (labels)
    targets: Vec<Arc<Tensor<CpuBackend, DenseStorage<Int32>, Int32>>>,
    /// Length of the dataset (samples in first dimension)
    length: usize,
}

impl TensorDataset {
    /// Creates a new TensorDataset from input and target tensors
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors (features, Float32 tensors)
    /// * `targets` - Vector of target tensors (labels, Int32 tensors)
    ///
    /// # Returns
    /// A TensorDataset, or an error if tensor dimensions are incompatible
    ///
    /// # Errors
    /// Returns `DataError::InvalidConfiguration` if:
    /// - Any tensor is empty
    /// - Input and target tensors have different lengths
    /// - Tensors have incompatible shapes for indexing
    pub fn new(
        inputs: Vec<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>,
        targets: Vec<Tensor<CpuBackend, DenseStorage<Int32>, Int32>>,
    ) -> Result<Self> {
        if inputs.is_empty() && targets.is_empty() {
            return Err(DataError::invalid_configuration(
                "At least one input or target tensor must be provided",
            ));
        }

        // Check that all tensors have compatible lengths
        let mut length = None;

        for tensor in &inputs {
            if tensor.is_empty() {
                return Err(DataError::invalid_configuration(
                    "Input tensors cannot be empty",
                ));
            }
            let tensor_len = tensor.shape().dims()[0];
            if let Some(len) = length {
                if len != tensor_len {
                    return Err(DataError::invalid_configuration(
                        "All input tensors must have the same length in first dimension",
                    ));
                }
            } else {
                length = Some(tensor_len);
            }
        }

        for tensor in &targets {
            if tensor.is_empty() {
                return Err(DataError::invalid_configuration(
                    "Target tensors cannot be empty",
                ));
            }
            let tensor_len = tensor.shape().dims()[0];
            if let Some(len) = length {
                if len != tensor_len {
                    return Err(DataError::invalid_configuration(
                        "All target tensors must have the same length in first dimension",
                    ));
                }
            } else {
                length = Some(tensor_len);
            }
        }

        let length = length.unwrap_or(0);

        Ok(Self {
            inputs: inputs.into_iter().map(Arc::new).collect(),
            targets: targets.into_iter().map(Arc::new).collect(),
            length,
        })
    }

    /// Creates a TensorDataset from inputs only (unsupervised learning)
    pub fn from_inputs(
        inputs: Vec<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>,
    ) -> Result<Self> {
        Self::new(inputs, vec![])
    }

    /// Creates a TensorDataset from targets only
    pub fn from_targets(
        targets: Vec<Tensor<CpuBackend, DenseStorage<Int32>, Int32>>,
    ) -> Result<Self> {
        Self::new(vec![], targets)
    }

    /// Returns the number of input tensors
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of target tensors
    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Returns a reference to the input tensors
    pub fn inputs(&self) -> &[Arc<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>] {
        &self.inputs
    }

    /// Returns a reference to the target tensors
    pub fn targets(&self) -> &[Arc<Tensor<CpuBackend, DenseStorage<Int32>, Int32>>] {
        &self.targets
    }
}

/// Sample type for TensorDataset
///
/// A sample consists of clones of the input and target tensors at a specific index.
#[derive(Debug, Clone)]
pub struct TensorSample {
    /// Input tensors for this sample
    pub inputs: Vec<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>,
    /// Target tensors for this sample
    pub targets: Vec<Tensor<CpuBackend, DenseStorage<Int32>, Int32>>,
}

impl Dataset<TensorSample> for TensorDataset {
    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<TensorSample> {
        if index >= self.length {
            return Err(DataError::index_out_of_bounds(index, self.length));
        }

        let mut inputs = Vec::with_capacity(self.inputs.len());
        let mut targets = Vec::with_capacity(self.targets.len());

        // Extract sample from each input tensor
        for tensor in &self.inputs {
            // For now, assume 1D tensors and get the element at index
            let slice = tensor.as_slice();
            if index >= slice.len() {
                return Err(DataError::index_out_of_bounds(index, slice.len()));
            }
            // Create a new 1-element tensor with the sample
            let sample_data = vec![slice[index]];
            let sample =
                Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(sample_data, &[1])?;
            inputs.push(sample);
        }

        // Extract sample from each target tensor
        for tensor in &self.targets {
            let slice = tensor.as_slice();
            if index >= slice.len() {
                return Err(DataError::index_out_of_bounds(index, slice.len()));
            }
            // Create a new 1-element tensor with the sample
            let sample_data = vec![slice[index]];
            let sample =
                Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(sample_data, &[1])?;
            targets.push(sample);
        }

        Ok(TensorSample { inputs, targets })
    }

    fn name(&self) -> &str {
        "TensorDataset"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_tensor::Tensor;

    #[test]
    fn test_tensor_dataset_creation() {
        let data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[4],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1), Int32::new(0), Int32::new(1)],
            &[4],
        )
        .unwrap();

        let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.num_inputs(), 1);
        assert_eq!(dataset.num_targets(), 1);
    }

    #[test]
    fn test_tensor_dataset_sample_access() {
        let data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[4],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1), Int32::new(0), Int32::new(1)],
            &[4],
        )
        .unwrap();

        let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();

        // Test first sample
        let sample = dataset.get(0).unwrap();
        assert_eq!(sample.inputs.len(), 1);
        assert_eq!(sample.targets.len(), 1);

        // Check values (should be scalar tensors)
        assert_eq!(sample.inputs[0].shape().dims(), &[1]);
        assert_eq!(sample.targets[0].shape().dims(), &[1]);

        // Test last sample
        let sample = dataset.get(3).unwrap();
        assert_eq!(sample.inputs[0].shape().dims(), &[1]);
        assert_eq!(sample.targets[0].shape().dims(), &[1]);
    }

    #[test]
    fn test_tensor_dataset_multiple_inputs() {
        let input1 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let input2 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0), Float32::new(20.0)],
            &[2],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let dataset = TensorDataset::new(vec![input1, input2], vec![targets]).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.num_inputs(), 2);
        assert_eq!(dataset.num_targets(), 1);

        let sample = dataset.get(0).unwrap();
        assert_eq!(sample.inputs.len(), 2);
        assert_eq!(sample.targets.len(), 1);
    }

    #[test]
    fn test_tensor_dataset_empty_inputs() {
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1), Int32::new(2)],
            &[3],
        )
        .unwrap();
        let dataset = TensorDataset::from_targets(vec![targets]).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.num_inputs(), 0);
        assert_eq!(dataset.num_targets(), 1);
    }

    #[test]
    fn test_tensor_dataset_empty_targets() {
        let inputs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();
        let dataset = TensorDataset::from_inputs(vec![inputs]).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.num_inputs(), 1);
        assert_eq!(dataset.num_targets(), 0);
    }

    #[test]
    fn test_tensor_dataset_mismatched_lengths() {
        let data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let result = TensorDataset::new(vec![data], vec![targets]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_dataset_empty_tensor() {
        let data = Tensor::from_vec(vec![], &[0]).unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let result = TensorDataset::new(vec![data], vec![targets]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_dataset_index_out_of_bounds() {
        let data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();

        assert!(dataset.get(2).is_err());
        assert!(dataset.get(10).is_err());
    }
}
