//! Thread safety tests for Tensor operations
//!
//! These tests verify that the Arc<RwLock> migration enables safe concurrent
//! tensor operations and parallel data loading.

use crate::Tensor;
use std::sync::Arc;
use std::thread;

/// Test concurrent read access to tensor gradients
#[test]
fn test_concurrent_gradient_reads() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor = Arc::new(tensor);

    // Set a gradient
    let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);
    tensor.set_grad(grad).unwrap();

    let mut handles = vec![];

    // Spawn multiple threads to read the gradient concurrently
    for _ in 0..4 {
        let tensor_clone = Arc::clone(&tensor);
        let handle = thread::spawn(move || {
            let grad = tensor_clone.grad();
            assert!(grad.is_some());
            let grad_data = grad.unwrap().data();
            assert_eq!(grad_data.len(), 3);
            assert_eq!(grad_data[0], 0.1);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test concurrent write access to tensor gradients (should be safe)
#[test]
fn test_concurrent_gradient_writes() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor = Arc::new(tensor);

    let mut handles = vec![];

    // Spawn multiple threads to write gradients
    for i in 0..4 {
        let tensor_clone = Arc::clone(&tensor);
        let handle = thread::spawn(move || {
            let grad = Tensor::from_vec(vec![i as f32 * 0.1; 3], vec![3]);
            tensor_clone.set_grad(grad).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify gradient was set (last write wins due to RwLock fairness)
    let final_grad = tensor.grad();
    assert!(final_grad.is_some());
}

/// Test that tensors can be sent across threads
#[test]
fn test_tensor_send_across_threads() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let handle = thread::spawn(move || {
        // Verify tensor works in the new thread
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.data()[0], 1.0);

        // Create a new tensor in this thread
        let new_tensor = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        assert_eq!(new_tensor.data()[1], 5.0);

        new_tensor
    });

    let result_tensor = handle.join().unwrap();
    assert_eq!(result_tensor.data()[2], 6.0);
}

/// Test concurrent tensor operations
#[test]
fn test_concurrent_tensor_operations() {
    let tensor1 = Arc::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]));
    let tensor2 = Arc::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]));

    let mut handles = vec![];

    // Spawn threads to perform operations concurrently
    for i in 0..4 {
        let t1 = Arc::clone(&tensor1);
        let t2 = Arc::clone(&tensor2);
        let handle = thread::spawn(move || {
            match i % 2 {
                0 => {
                    // Addition
                    let result = (t1.as_ref() + t2.as_ref()).unwrap();
                    assert_eq!(result.data()[0], 5.0);
                }
                1 => {
                    // Multiplication
                    let result = (t1.as_ref() * t2.as_ref()).unwrap();
                    assert_eq!(result.data()[0], 4.0);
                }
                _ => unreachable!(),
            }
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test that DataLoader can be used with Send + Sync datasets
#[test]
fn test_dataloader_thread_safety_bounds() {
    use coeus_utils::{DataLoader, Dataset};

    // Create a simple dataset
    struct SimpleDataset(Vec<Tensor<f32>>, Vec<Tensor<f32>>);

    impl Dataset<f32> for SimpleDataset {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
            (self.0[index].clone(), self.1[index].clone())
        }
    }

    let data = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    let targets = vec![Tensor::from_vec(vec![0.0], vec![1])];
    let dataset = SimpleDataset(data, targets);

    // Create DataLoader - this should work because Tensor is now Send + Sync
    let dataloader = DataLoader::builder(dataset)
        .batch_size(1)
        .num_workers(2)
        .build();

    // Verify DataLoader implements Send + Sync
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    assert_send_sync(&dataloader);

    // Verify it can iterate
    let batches: Vec<_> = dataloader.into_iter().collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].data.shape(), &[1, 2]);
}
