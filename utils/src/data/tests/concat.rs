//! Tests for ConcatDataset

use crate::{Tensor, TensorDataset, ConcatDataset, Dataset};

/// Mock dataset for testing
struct MockDataset {
    data: Vec<(Tensor<f64>, Tensor<f64>)>,
}

impl MockDataset {
    fn new(data: Vec<(Tensor<f64>, Tensor<f64>)>) -> Self {
        Self { data }
    }
}

impl Dataset<f64> for MockDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> (Tensor<f64>, Tensor<f64>) {
        self.data[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test ConcatDataset basic functionality
    #[test]
    fn test_concat_dataset_basic() {
        let data1 = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
        let targets1 = vec![Tensor::from_vec(vec![0.0], vec![1])];
        let dataset1 = Box::new(TensorDataset::new(data1, targets1));

        let data2 = vec![Tensor::from_vec(vec![3.0, 4.0], vec![2])];
        let targets2 = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let dataset2 = Box::new(TensorDataset::new(data2, targets2));

        let concat_dataset = ConcatDataset::new(vec![dataset1, dataset2]);

        assert_eq!(concat_dataset.len(), 2);

        let (sample1, target1) = concat_dataset.get(0);
        assert_eq!(sample1.data(), &[1.0, 2.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = concat_dataset.get(1);
        assert_eq!(sample2.data(), &[3.0, 4.0]);
        assert_eq!(target2.data(), &[1.0]);
    }

    /// Test ConcatDataset with multiple datasets
    #[test]
    fn test_concat_dataset_multiple() {
        let datasets: Vec<Box<dyn Dataset<f64>>> = vec![
            Box::new(TensorDataset::new(
                vec![Tensor::from_vec(vec![1.0], vec![1])],
                vec![Tensor::from_vec(vec![0.0], vec![1])],
            )),
            Box::new(TensorDataset::new(
                vec![Tensor::from_vec(vec![2.0], vec![1])],
                vec![Tensor::from_vec(vec![1.0], vec![1])],
            )),
            Box::new(TensorDataset::new(
                vec![Tensor::from_vec(vec![3.0], vec![1])],
                vec![Tensor::from_vec(vec![2.0], vec![1])],
            )),
        ];

        let concat_dataset = ConcatDataset::new(datasets);
        assert_eq!(concat_dataset.len(), 3);

        let (sample1, target1) = concat_dataset.get(0);
        assert_eq!(sample1.data(), &[1.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = concat_dataset.get(1);
        assert_eq!(sample2.data(), &[2.0]);
        assert_eq!(target2.data(), &[1.0]);

        let (sample3, target3) = concat_dataset.get(2);
        assert_eq!(sample3.data(), &[3.0]);
        assert_eq!(target3.data(), &[2.0]);
    }

    /// Test ConcatDataset edge cases
    #[test]
    fn test_concat_dataset_edge_cases() {
        // Test with empty dataset list
        let empty_datasets: Vec<Box<dyn Dataset<f64>>> = vec![];
        let empty_concat = ConcatDataset::new(empty_datasets);
        assert_eq!(empty_concat.len(), 0);
        assert!(empty_concat.is_empty());

        // Test with single dataset
        let single_dataset = Box::new(TensorDataset::new(
            vec![Tensor::from_vec(vec![42.0], vec![1])],
            vec![Tensor::from_vec(vec![0.0], vec![1])],
        ));
        let single_concat = ConcatDataset::new(vec![single_dataset]);
        assert_eq!(single_concat.len(), 1);
        let (sample, target) = single_concat.get(0);
        assert_eq!(sample.data(), &[42.0]);
        assert_eq!(target.data(), &[0.0]);
    }

    /// Test ConcatDataset with different dataset types
    #[test]
    fn test_concat_dataset_mixed_types() {
        let data1 = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
        let targets1 = vec![Tensor::from_vec(vec![0.0], vec![1])];
        let dataset1 = Box::new(TensorDataset::new(data1, targets1));

        // Create a mock dataset with different data shape
        let mock_data = vec![(
            Tensor::from_vec(vec![10.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        )];
        let mock_dataset: Box<dyn Dataset<f64>> = Box::new(MockDataset::new(mock_data));

        let datasets = vec![dataset1, mock_dataset];
        let concat_dataset = ConcatDataset::new(datasets);

        assert_eq!(concat_dataset.len(), 2);

        let (sample1, target1) = concat_dataset.get(0);
        assert_eq!(sample1.data(), &[1.0, 2.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = concat_dataset.get(1);
        assert_eq!(sample2.data(), &[10.0]);
        assert_eq!(target2.data(), &[1.0]);
    }
}
