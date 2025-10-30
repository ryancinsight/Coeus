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

// Type aliases to reduce type complexity
type Float32Tensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
type Int32Tensor = Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>;

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
/// let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)], &[4]).unwrap();
/// let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(vec![Int32::new(0), Int32::new(1), Int32::new(0), Int32::new(1)], &[4]).unwrap();
///
/// // Create dataset
/// let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();
/// assert_eq!(dataset.len(), 4);
///
/// // Get first sample
/// let sample = dataset.get(0).unwrap();
/// // sample contains tensors for data and targets
/// ```
#[derive(Clone)]
pub struct TensorDataset {
    /// Input tensors (features)
    inputs: Vec<Arc<Float32Tensor>>,
    /// Target tensors (labels)
    targets: Vec<Arc<Int32Tensor>>,
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
        inputs: Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
        targets: Vec<Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>>,
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
        inputs: Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
    ) -> Result<Self> {
        Self::new(inputs, vec![])
    }

    /// Creates a TensorDataset from targets only
    pub fn from_targets(
        targets: Vec<Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>>,
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
    pub fn inputs(&self) -> &[Arc<Float32Tensor>] {
        &self.inputs
    }

    /// Returns a reference to the target tensors
    pub fn targets(&self) -> &[Arc<Int32Tensor>] {
        &self.targets
    }
}

/// Sample type for TensorDataset
///
/// A sample consists of clones of the input and target tensors at a specific index.
#[derive(Debug, Clone)]
pub struct TensorSample {
    /// Input tensors for this sample
    pub inputs: Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
    /// Target tensors for this sample
    pub targets: Vec<Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>>,
}

/// A dataset that concatenates multiple datasets into a single dataset
///
/// This allows training on data from multiple sources by combining them
/// into one contiguous dataset. The datasets are accessed in the order
/// they were provided to the constructor.
///
/// # Example
///
/// ```rust
/// use coeus_utils::{Dataset, TensorDataset, ConcatDataset};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::{float::Float32, int::Int32};
///
/// // Create sample data for first dataset
/// let inputs1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)],
///     &[2]
/// ).unwrap();
/// let targets1 = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
///     vec![Int32::new(0), Int32::new(1)],
///     &[2]
/// ).unwrap();
///
/// // Create sample data for second dataset
/// let inputs2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(3.0), Float32::new(4.0)],
///     &[2]
/// ).unwrap();
/// let targets2 = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
///     vec![Int32::new(2), Int32::new(3)],
///     &[2]
/// ).unwrap();
///
/// // Create two separate datasets
/// let dataset1 = TensorDataset::new(vec![inputs1], vec![targets1]).unwrap();
/// let dataset2 = TensorDataset::new(vec![inputs2], vec![targets2]).unwrap();
///
/// // Combine them into one dataset
/// let combined = ConcatDataset::new(vec![dataset1, dataset2]).unwrap();
/// assert_eq!(combined.len(), 4); // Each dataset has 2 samples
///
/// // Access samples from both original datasets
/// let sample1 = combined.get(0).unwrap();      // From dataset1
/// let sample2 = combined.get(1).unwrap();      // From dataset2
/// ```
#[derive(Clone)]
pub struct ConcatDataset {
    /// The datasets to concatenate
    datasets: Vec<TensorDataset>,
    /// Cumulative lengths for efficient indexing
    cumulative_lengths: Vec<usize>,
    /// Total length of all datasets combined
    total_length: usize,
}

impl ConcatDataset {
    /// Creates a new ConcatDataset from a vector of TensorDatasets
    ///
    /// # Arguments
    /// * `datasets` - Vector of TensorDatasets to concatenate
    ///
    /// # Returns
    /// A ConcatDataset containing all datasets, or an error if datasets is empty
    ///
    /// # Errors
    /// Returns `DataError::InvalidConfiguration` if `datasets` is empty
    pub fn new(datasets: Vec<TensorDataset>) -> Result<Self> {
        if datasets.is_empty() {
            return Err(DataError::invalid_configuration(
                "At least one dataset must be provided for concatenation",
            ));
        }

        let mut cumulative_lengths = Vec::with_capacity(datasets.len());
        let mut running_total = 0;

        for dataset in &datasets {
            running_total += dataset.len();
            cumulative_lengths.push(running_total);
        }

        let total_length = running_total;

        Ok(Self {
            datasets,
            cumulative_lengths,
            total_length,
        })
    }

    /// Returns the number of constituent datasets
    pub fn num_datasets(&self) -> usize {
        self.datasets.len()
    }

    /// Returns a reference to the constituent datasets
    pub fn datasets(&self) -> &[TensorDataset] {
        &self.datasets
    }
}

impl Dataset<TensorSample> for ConcatDataset {
    fn len(&self) -> usize {
        self.total_length
    }

    fn get(&self, index: usize) -> Result<TensorSample> {
        if index >= self.total_length {
            return Err(DataError::index_out_of_bounds(index, self.total_length));
        }

        // Find which dataset this index belongs to using binary search
        // on the cumulative lengths
        let dataset_index = match self.cumulative_lengths.binary_search(&index) {
            Ok(idx) => {
                // Index exactly matches the end of a dataset
                // This means we should access the last element of dataset at idx
                idx
            }
            Err(idx) => {
                // Index falls within dataset at idx
                idx
            }
        };

        if dataset_index >= self.datasets.len() {
            // This shouldn't happen given the bounds check above, but handle it
            return Err(DataError::index_out_of_bounds(index, self.total_length));
        }

        let dataset = &self.datasets[dataset_index];
        let offset = if dataset_index == 0 {
            index
        } else {
            index - self.cumulative_lengths[dataset_index - 1]
        };

        dataset.get(offset)
    }

    fn name(&self) -> &str {
        "ConcatDataset"
    }
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
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(sample_data, &[1])?;
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
                Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(sample_data, &[1])?;
            targets.push(sample);
        }

        Ok(TensorSample { inputs, targets })
    }

    fn name(&self) -> &str {
        "TensorDataset"
    }
}

/// A dataset that creates a subset of another dataset using given indices
///
/// This allows for flexible dataset sampling, such as creating training/validation
/// splits or sampling specific examples from a larger dataset.
///
/// # Example
///
/// ```rust
/// use coeus_utils::{Dataset, TensorDataset, Subset};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::{float::Float32, int::Int32};
///
/// // Create sample data
/// let inputs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
/// let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
///     vec![Int32::new(0), Int32::new(1), Int32::new(0)],
///     &[3]
/// ).unwrap();
///
/// // Create a dataset
/// let dataset = TensorDataset::new(vec![inputs], vec![targets]).unwrap();
///
/// // Create a subset with indices [0, 2, 4] (note: dataset only has 3 samples)
/// let subset = Subset::new(dataset, vec![0, 2]).unwrap();
/// assert_eq!(subset.len(), 2);
///
/// // Access subset samples
/// let sample = subset.get(0).unwrap(); // Gets original sample at index 0
/// ```
#[derive(Clone)]
pub struct Subset {
    /// The original dataset
    dataset: TensorDataset,
    /// Indices mapping subset indices to original dataset indices
    indices: Vec<usize>,
}

impl Subset {
    /// Creates a new Subset from a dataset and indices
    ///
    /// # Arguments
    /// * `dataset` - The original dataset to create a subset from
    /// * `indices` - Vector of indices from the original dataset
    ///
    /// # Returns
    /// A Subset dataset, or an error if any index is out of bounds
    ///
    /// # Errors
    /// Returns `DataError::IndexOutOfBounds` if any index is >= dataset.len()
    pub fn new(dataset: TensorDataset, indices: Vec<usize>) -> Result<Self> {
        // Validate all indices are within bounds
        let dataset_len = dataset.len();
        for &index in &indices {
            if index >= dataset_len {
                return Err(DataError::index_out_of_bounds(index, dataset_len));
            }
        }

        Ok(Self { dataset, indices })
    }

    /// Returns the indices used to create this subset
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns a reference to the original dataset
    pub fn dataset(&self) -> &TensorDataset {
        &self.dataset
    }
}

impl Dataset<TensorSample> for Subset {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<TensorSample> {
        if index >= self.indices.len() {
            return Err(DataError::index_out_of_bounds(index, self.indices.len()));
        }

        let original_index = self.indices[index];
        self.dataset.get(original_index)
    }

    fn name(&self) -> &str {
        "Subset"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_tensor::Tensor;

    #[test]
    fn test_tensor_dataset_creation() {
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[4],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
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
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[4],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
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
        let input1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let input2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0), Float32::new(20.0)],
            &[2],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
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
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
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
        let inputs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
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
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
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
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let result = TensorDataset::new(vec![data], vec![targets]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_dataset_index_out_of_bounds() {
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
            vec![Int32::new(0), Int32::new(1)],
            &[2],
        )
        .unwrap();

        let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();

        assert!(dataset.get(2).is_err());
        assert!(dataset.get(10).is_err());
    }
}
