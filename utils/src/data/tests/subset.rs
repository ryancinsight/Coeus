//! Tests for Subset

use crate::{Tensor, TensorDataset, Subset, Dataset};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Subset dataset functionality
    #[test]
    fn test_subset_basic() {
        let data = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![2.0], vec![1]),
        ];

        let full_dataset = TensorDataset::new(data, targets);
        let subset_indices = vec![0, 2];
        let subset = Subset::new(full_dataset, subset_indices);

        assert_eq!(subset.len(), 2);

        let (sample1, target1) = subset.get(0);
        assert_eq!(sample1.data(), &[1.0, 2.0]);
        assert_eq!(target1.data(), &[0.0]);

        let (sample2, target2) = subset.get(1);
        assert_eq!(sample2.data(), &[5.0, 6.0]);
        assert_eq!(target2.data(), &[2.0]);
    }

    /// Test Subset edge cases
    #[test]
    fn test_subset_edge_cases() {
        let data = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let targets = vec![Tensor::from_vec(vec![0.0], vec![1])];

        let full_dataset = TensorDataset::new(data, targets);

        // Empty subset
        let empty_subset = Subset::new(full_dataset.clone(), vec![]);
        assert_eq!(empty_subset.len(), 0);
        assert!(empty_subset.is_empty());

        // Single element subset
        let single_subset = Subset::new(full_dataset, vec![0]);
        assert_eq!(single_subset.len(), 1);
        let (sample, target) = single_subset.get(0);
        assert_eq!(sample.data(), &[1.0]);
        assert_eq!(target.data(), &[0.0]);
    }
}
