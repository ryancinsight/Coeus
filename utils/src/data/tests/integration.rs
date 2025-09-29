//! Integration tests for dataset functionality

use crate::{Tensor, TensorDataset, Dataset, CpuBackend};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test error handling and edge cases
    #[test]
    fn test_dataset_error_handling() {
        let data = vec![Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1])];
        let targets = vec![Tensor::from_vec(CpuBackend::new(), vec![0.0], vec![1])];
        let dataset = TensorDataset::new(data, targets);

        // Test out of bounds access
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dataset.get(10)));
        // Should panic due to index out of bounds
        assert!(result.is_err());
    }

    /// Test large dataset performance
    #[test]
    fn test_dataset_performance() {
        // Create a large dataset
        let size = 10000;
        let data: Vec<Tensor<f64>> = (0..size)
            .map(|i| Tensor::from_vec(CpuBackend::new(), vec![i as f64], vec![1]))
            .collect();
        let targets: Vec<Tensor<f64>> = (0..size)
            .map(|i| Tensor::from_vec(CpuBackend::new(), vec![(i % 2) as f64], vec![1]))
            .collect();

        let dataset = TensorDataset::new(data, targets);

        assert_eq!(dataset.len(), size);

        // Test random access
        let (sample, target) = dataset.get(size / 2);
        assert_eq!(sample.data()[0], (size / 2) as f64);
        assert_eq!(target.data()[0], ((size / 2) % 2) as f64);

        // Test iteration
        let mut count = 0;
        for (sample, target) in dataset.iter() {
            assert_eq!(sample.data()[0], count as f64);
            assert_eq!(target.data()[0], (count % 2) as f64);
            count += 1;
        }
        assert_eq!(count, size);
    }

    /// Test memory safety with large datasets
    #[test]
    fn test_dataset_memory_safety() {
        let data: Vec<Tensor<f64>> = (0..1000)
            .map(|i| Tensor::from_vec(CpuBackend::new(), vec![i as f64; 100], vec![100]))
            .collect();
        let targets: Vec<Tensor<f64>> = (0..1000)
            .map(|i| Tensor::from_vec(CpuBackend::new(), vec![(i % 10) as f64], vec![1]))
            .collect();

        let dataset = TensorDataset::new(data, targets);

        // Test that we can access elements without memory corruption
        let (sample, target) = dataset.get(999);
        assert_eq!(sample.data()[0], 999.0);
        assert_eq!(target.data()[0], 9.0);

        // Test iteration
        for (count, (sample, target)) in dataset.iter().take(10).enumerate() {
            assert_eq!(sample.data()[0], count as f64);
            assert_eq!(target.data()[0], (count % 10) as f64);
        }
    }
}
