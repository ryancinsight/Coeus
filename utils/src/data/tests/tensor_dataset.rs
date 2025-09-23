//! Tests for TensorDataset

use crate::{Tensor, TensorDataset, Dataset};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test TensorDataset basic functionality
    #[test]
    fn test_tensor_dataset_basic() {
        let data = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];

        let dataset = TensorDataset::new(data, targets);

        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());

        let (sample1, target1) = dataset.get(0);
        assert_eq!(sample1.data(), &[1.0, 2.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = dataset.get(1);
        assert_eq!(sample2.data(), &[3.0, 4.0]);
        assert_eq!(target2.data(), &[1.0]);
    }

    /// Test TensorDataset edge cases
    #[test]
    fn test_tensor_dataset_edge_cases() {
        // Test empty dataset
        let empty_data: Vec<Tensor<f64>> = vec![];
        let empty_targets: Vec<Tensor<f64>> = vec![];
        let empty_dataset = TensorDataset::new(empty_data, empty_targets);

        assert_eq!(empty_dataset.len(), 0);
        assert!(empty_dataset.is_empty());

        // Test single element dataset
        let single_data = vec![Tensor::from_vec(vec![42.0], vec![1])];
        let single_targets = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let single_dataset = TensorDataset::new(single_data, single_targets);

        assert_eq!(single_dataset.len(), 1);
        let (sample, target) = single_dataset.get(0);
        assert_eq!(sample.data(), &[42.0]);
        assert_eq!(target.data(), &[1.0]);

        // Test from_arrays method
        let data_array = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let targets_array = vec![Tensor::from_vec(vec![0.0], vec![1])];
        let array_dataset = TensorDataset::from_arrays(&data_array, &targets_array);

        assert_eq!(array_dataset.len(), 1);
        let (sample, target) = array_dataset.get(0);
        assert_eq!(sample.data(), &[1.0]);
        assert_eq!(target.data(), &[0.0]);
    }
}
