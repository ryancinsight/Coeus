//! Tests for DataLoader functionality
//!
//! This module tests the DataLoader implementation to ensure it properly
//! stacks tensors and creates batches as expected.

use coeus_tensor::Tensor;
use coeus_utils::utils::tensor_ops::stack;
use coeus_utils::{DataLoader, Dataset};

/// Test the tensor stacking functionality directly
#[test]
fn test_tensor_stacking() {
    // Create some sample tensors
    let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor2 = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

    // Stack them along dimension 0
    let stacked = stack(&[&tensor1, &tensor2], 0).unwrap();

    // Verify the result
    assert_eq!(stacked.shape(), &[2, 3]);
    let data = stacked.data();
    assert_eq!(data[0], 1.0); // First tensor, first element
    assert_eq!(data[1], 2.0); // First tensor, second element
    assert_eq!(data[2], 3.0); // First tensor, third element
    assert_eq!(data[3], 4.0); // Second tensor, first element
    assert_eq!(data[4], 5.0); // Second tensor, second element
    assert_eq!(data[5], 6.0); // Second tensor, third element
}

/// Test stacking with different dimensions
#[test]
fn test_tensor_stacking_different_dims() {
    // Create tensors with different shapes for stacking
    let tensor1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

    // Stack along dimension 0 (batch dimension)
    let stacked = stack(&[&tensor1, &tensor2], 0).unwrap();
    assert_eq!(stacked.shape(), &[2, 2]);

    // Stack along dimension 1 (feature dimension)
    let stacked_dim1 = stack(&[&tensor1, &tensor2], 1).unwrap();
    assert_eq!(stacked_dim1.shape(), &[2, 2]);
}

/// Test stacking empty tensor list (should fail)
#[test]
fn test_tensor_stacking_empty() {
    let result = stack::<f32>(&[], 0);
    assert!(result.is_err());
}

/// Test stacking tensors with mismatched shapes (should fail)
#[test]
fn test_tensor_stacking_mismatched_shapes() {
    let tensor1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]); // Different shape

    let result = stack(&[&tensor1, &tensor2], 0);
    assert!(result.is_err());
}

/// Dataset for DataLoader testing
struct DatasetImpl {
    data: Vec<Tensor<f32>>,
    targets: Vec<Tensor<f32>>,
}

impl DatasetImpl {
    fn new(size: usize) -> Self {
        let mut data = Vec::new();
        let mut targets = Vec::new();

        for i in 0..size {
            // Create data tensor with shape [3] (e.g., 3 features)
            let data_tensor =
                Tensor::from_vec(vec![i as f32, (i + 1) as f32, (i + 2) as f32], vec![3]);
            data.push(data_tensor);

            // Create target tensor with shape [1] (e.g., single target)
            let target_tensor = Tensor::from_vec(vec![(i % 2) as f32], vec![1]);
            targets.push(target_tensor);
        }

        Self { data, targets }
    }
}

impl Dataset<f32> for DatasetImpl {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
        (self.data[index].clone(), self.targets[index].clone())
    }
}

/// Test parallel data loading functionality
#[test]
fn test_dataloader_parallel_loading() {
    // Create a larger test dataset
    let dataset = DatasetImpl::new(12);

    // Create DataLoader with multiple workers
    let dataloader = DataLoader::builder(dataset)
        .batch_size(3)
        .shuffle(false)
        .num_workers(2) // Use 2 workers for parallel loading
        .build();

    // Collect all batches
    let batches: Vec<_> = dataloader.into_iter().collect();

    // Should have 4 batches (12 samples / 3 batch_size)
    assert_eq!(batches.len(), 4);

    // Verify all batches have correct structure
    for batch in batches {
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.data.shape(), &[3, 3]); // [batch_size, features]
        assert_eq!(batch.targets.shape(), &[3, 1]); // [batch_size, targets]
    }
}

/// Test that DataLoader works with Send + Sync datasets
#[test]
fn test_dataloader_send_sync_bounds() {
    // This test verifies that our DataLoader works with Send + Sync bounds
    // which are required for parallel operations
    let dataset = DatasetImpl::new(6);

    let dataloader = DataLoader::builder(dataset).batch_size(2).build();

    // Verify it implements Send + Sync (compile-time check)
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    assert_send_sync(&dataloader);

    // Verify functionality still works
    let batches: Vec<_> = dataloader.into_iter().collect();
    assert_eq!(batches.len(), 3);
}

/// Test DataLoader with empty dataset
#[test]
fn test_dataloader_empty_dataset() {
    struct EmptyDataset;
    impl Dataset<f32> for EmptyDataset {
        fn len(&self) -> usize {
            0
        }
        fn get(&self, _index: usize) -> (Tensor<f32>, Tensor<f32>) {
            panic!("Should not be called")
        }
    }

    let dataset = EmptyDataset;
    let dataloader = DataLoader::builder(dataset).batch_size(1).build();

    // Should handle empty dataset gracefully
    let batches: Vec<_> = dataloader.into_iter().collect();
    assert_eq!(batches.len(), 0);
}

/// Test DataLoader with batch size larger than dataset
#[test]
fn test_dataloader_batch_larger_than_dataset() {
    let dataset = DatasetImpl::new(3);
    let dataloader = DataLoader::builder(dataset).batch_size(10).build();

    let batches: Vec<_> = dataloader.into_iter().collect();
    assert_eq!(batches.len(), 1);

    let batch = &batches[0];
    assert_eq!(batch.data.shape()[0], 3); // Should have all 3 samples
}

/// Test DataLoader error handling for tensor operations
#[test]
fn test_dataloader_tensor_operation_errors() {
    // Create dataset with incompatible tensor shapes
    struct IncompatibleDataset;
    impl Dataset<f32> for IncompatibleDataset {
        fn len(&self) -> usize {
            2
        }
        fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
            match index {
                0 => (
                    Tensor::from_vec(vec![1.0, 2.0], vec![2]),
                    Tensor::from_vec(vec![0.0], vec![1]),
                ),
                1 => (
                    Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]),
                    Tensor::from_vec(vec![1.0], vec![1]),
                ), // Different shape
                _ => panic!("Invalid index"),
            }
        }
    }

    let dataset = IncompatibleDataset;
    let dataloader = DataLoader::builder(dataset).batch_size(2).build();

    // This test expects the DataLoader to panic on incompatible shapes
    // In production, this should be handled more gracefully
    let result = std::panic::catch_unwind(|| {
        let _batches: Vec<_> = dataloader.into_iter().collect();
    });
    // The operation should panic due to shape incompatibility
    assert!(result.is_err());
}

/// Test DataLoader with parallel loading and tracking
#[test]
fn test_dataloader_parallel_loading_tracking() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    // Create a dataset that tracks access
    #[derive(Clone)]
    struct TrackingDataset {
        access_count: Arc<Mutex<usize>>,
    }

    impl TrackingDataset {
        fn new() -> Self {
            Self {
                access_count: Arc::new(Mutex::new(0)),
            }
        }
    }

    impl Dataset<f32> for TrackingDataset {
        fn len(&self) -> usize {
            10
        }
        fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
            // Simulate some work
            thread::sleep(Duration::from_millis(1));

            let mut count = self.access_count.lock().unwrap();
            *count += 1;

            (
                Tensor::from_vec(vec![index as f32], vec![1]),
                Tensor::scalar(index as f32),
            )
        }
    }

    let dataset = TrackingDataset::new();
    let _initial_count = *dataset.access_count.lock().unwrap();

    let dataloader = DataLoader::builder(dataset)
        .batch_size(2)
        .num_workers(2)
        .build();

    let batches: Vec<_> = dataloader.into_iter().collect();

    // Verify we got the expected number of batches
    assert_eq!(batches.len(), 5); // 10 samples / 2 batch_size = 5 batches
}

/// Test DataLoader with different tensor shapes in same batch
#[test]
fn test_dataloader_mixed_shapes() {
    struct MixedShapeDataset;
    impl Dataset<f32> for MixedShapeDataset {
        fn len(&self) -> usize {
            3
        }
        fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
            match index {
                0 => (Tensor::from_vec(vec![1.0], vec![1]), Tensor::scalar(0.0)),
                1 => (
                    Tensor::from_vec(vec![2.0, 3.0], vec![2]),
                    Tensor::scalar(1.0),
                ),
                2 => (
                    Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]),
                    Tensor::scalar(2.0),
                ),
                _ => panic!("Invalid index"),
            }
        }
    }

    let dataset = MixedShapeDataset;
    let dataloader = DataLoader::builder(dataset).batch_size(3).build();

    // This test expects the DataLoader to panic on incompatible shapes
    // In production, this should be handled more gracefully
    let result = std::panic::catch_unwind(|| {
        let _batches: Vec<_> = dataloader.into_iter().collect();
    });
    // The operation should panic due to shape incompatibility
    assert!(result.is_err());
}
