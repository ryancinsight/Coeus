//! Tests for DatasetIter

use crate::{Tensor, TensorDataset, Dataset};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test DatasetIter functionality
    #[test]
    fn test_dataset_iter() {
        let data = vec![
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![2.0], vec![1]),
            Tensor::from_vec(vec![3.0], vec![1]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![2.0], vec![1]),
        ];

        let dataset = TensorDataset::new(data, targets);
        let mut iter = dataset.iter();

        assert_eq!(iter.len(), 3);

        let (sample1, target1) = iter.next().unwrap();
        assert_eq!(sample1.data(), &[1.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = iter.next().unwrap();
        assert_eq!(sample2.data(), &[2.0]);
        assert_eq!(target2.data(), &[1.0]);

        let (sample3, target3) = iter.next().unwrap();
        assert_eq!(sample3.data(), &[3.0]);
        assert_eq!(target3.data(), &[2.0]);

        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }

    /// Test DatasetIter with empty dataset
    #[test]
    fn test_dataset_iter_empty() {
        let empty_data: Vec<Tensor<f64>> = vec![];
        let empty_targets: Vec<Tensor<f64>> = vec![];
        let empty_dataset = TensorDataset::new(empty_data, empty_targets);

        let mut iter = empty_dataset.iter();
        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }
}
