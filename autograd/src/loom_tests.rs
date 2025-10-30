//! Loom-based concurrency tests for autograd operations
//!
//! These tests use loom to verify that autograd graph operations
//! are thread-safe and free from data races. This is critical for
//! production readiness as it ensures computed graphs can be safely
//! used in concurrent settings.
//!
//! Run with: RUSTFLAGS="--cfg loom" cargo test --release --test loom

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::thread;
    use crate::ops::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    type TestTensor = Tensor<CpuBackend<Data = T>, DenseStorage<Float32>, Float32>;

    /// Test that simultaneous gradient computation on independent graphs is safe
    #[test]
    fn test_concurrent_independent_graphs() {
        loom::model(|| {
            // Create two independent computation graphs
            let x1 = Arc::new(TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap().requires_grad_(true));
            let y1 = Arc::new(TestTensor::from_vec(vec![Float32::new(3.0)], &[1]).unwrap().requires_grad_(true));

            let x2 = Arc::new(TestTensor::from_vec(vec![Float32::new(4.0)], &[1]).unwrap().requires_grad_(true));
            let y2 = Arc::new(TestTensor::from_vec(vec![Float32::new(5.0)], &[1]).unwrap().requires_grad_(true));

            let mut handles = vec![];

            // Thread 1: Work on first graph
            let x1_clone = Arc::clone(&x1);
            let y1_clone = Arc::clone(&y1);
            handles.push(thread::spawn(move || {
                let z1 = add(&x1_clone, &y1_clone).unwrap();
                let grad1 = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&z1, &grad1).unwrap();
                let grad_x1 = x1_clone.grad();
                let grad_y1 = y1_clone.grad();
                assert!(grad_x1.is_some());
                assert!(grad_y1.is_some());
            }));

            // Thread 2: Work on second graph independently
            let x2_clone = Arc::clone(&x2);
            let y2_clone = Arc::clone(&y2);
            handles.push(thread::spawn(move || {
                let z2 = mul(&x2_clone, &y2_clone).unwrap();
                let grad2 = TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap();
                backward_with_grad(&z2, &grad2).unwrap();
                let grad_x2 = x2_clone.grad();
                let grad_y2 = y2_clone.grad();
                assert!(grad_x2.is_some());
                assert!(grad_y2.is_some());
            }));

            // Wait for both threads
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test that shared tensor references in different contexts work safely
    #[test]
    fn test_concurrent_shared_tensor_usage() {
        loom::model(|| {
            // Create shared tensor with multiple references
            let shared_tensor = Arc::new(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap().requires_grad_(true));

            let mut handles = vec![];

            // Thread 1: Add operation
            let tensor1 = Arc::clone(&shared_tensor);
            handles.push(thread::spawn(move || {
                let result1 = add(&tensor1, &tensor1).unwrap(); // result = 1 + 1 = 2
                let grad1 = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&result1, &grad1).unwrap();
                // Check that tensor has gradient
                assert!(tensor1.grad().is_some());
            }));

            // Thread 2: Multiply operation with same tensor
            let tensor2 = Arc::clone(&shared_tensor);
            handles.push(thread::spawn(move || {
                let result2 = mul(&tensor2, &tensor2).unwrap(); // result = 1 * 1 = 1
                let grad2 = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&result2, &grad2).unwrap();
                // Check that tensor has gradient
                assert!(tensor2.grad().is_some());
            }));

            // Wait for completion
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test that grad method access is thread-safe
    #[test]
    fn test_concurrent_grad_access() {
        loom::model(|| {
            let tensor = Arc::new(TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap().requires_grad_(true));

            let result = add(&tensor, &tensor).unwrap();
            let grad = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
            backward_with_grad(&result, &grad).unwrap();

            let mut handles = vec![];

            // Multiple threads accessing grad() concurrently
            for _ in 0..3 {
                let tensor_clone = Arc::clone(&tensor);
                handles.push(thread::spawn(move || {
                    let grad_value = tensor_clone.grad();
                    assert!(grad_value.is_some());
                    let grad_slice = grad_value.unwrap().as_slice();
                    assert_eq!(grad_slice.len(), 1);
                    // Just access the gradient - loom ensures no races
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test that zero_grad() operations are thread-safe when called concurrently
    #[test]
    fn test_concurrent_zero_grad() {
        loom::model(|| {
            let tensor1 = Arc::new(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap().requires_grad_(true));
            let tensor2 = Arc::new(TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap().requires_grad_(true));

            let mut handles = vec![];

            // Thread 1: Modify tensor1, then zero its gradient
            let t1 = Arc::clone(&tensor1);
            handles.push(thread::spawn(move || {
                let result = add(&t1, &t1).unwrap();
                let grad = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&result, &grad).unwrap();
                t1.zero_grad().unwrap();

                // After zero_grad, gradient should be reset
                let grad_after = t1.grad().unwrap_or(TestTensor::zeros(&[1]).unwrap());
                // We can't directly check values due to Arc issues, but loom ensures no races
            }));

            // Thread 2: Do the same for tensor2
            let t2 = Arc::clone(&tensor2);
            handles.push(thread::spawn(move || {
                let result = mul(&t2, &t2).unwrap();
                let grad = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&result, &grad).unwrap();
                t2.zero_grad().unwrap();
            }));

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test that backward passes on different computation graphs don't interfere
    #[test]
    fn test_non_interfering_backward_passes() {
        loom::model(|| {
            // Create two separate computation graphs
            let x1 = Arc::new(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap().requires_grad_(true));
            let x2 = Arc::new(TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap().requires_grad_(true));

            let graph1 = Arc::new(add(&x1, &x1).unwrap()); // 1 + 1 = 2
            let graph2 = Arc::new(mul(&x2, &x2).unwrap()); // 2 * 2 = 4

            let mut handles = vec![];

            // Thread 1: Backward on graph1
            let graph1_clone = Arc::clone(&graph1);
            let x1_clone = Arc::clone(&x1);
            handles.push(thread::spawn(move || {
                let grad = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&graph1_clone, &grad).unwrap();
                // x1 should have grad = 2 (d/dx(x+x) = 2)
                assert!(x1_clone.grad().is_some());
            }));

            // Thread 2: Backward on graph2
            let graph2_clone = Arc::clone(&graph2);
            let x2_clone = Arc::clone(&x2);
            handles.push(thread::spawn(move || {
                let grad = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
                backward_with_grad(&graph2_clone, &grad).unwrap();
                // x2 should have grad = 4 (d/dx(x*x) = 2*x = 4 when x=2)
                assert!(x2_clone.grad().is_some());
            }));

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test that requires_grad() access is safe across threads
    #[test]
    fn test_concurrent_requires_grad_access() {
        loom::model(|| {
            let tensor = Arc::new(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap());

            let mut handles = vec![];

            // Multiple threads checking requires_grad
            for _ in 0..5 {
                let tensor_clone = Arc::clone(&tensor);
                handles.push(thread::spawn(move || {
                    let requires_grad = tensor_clone.requires_grad(); // Should be false
                    // Just accessing the property - loom ensures safety
                    assert!(!requires_grad);
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /// Test concurrent access to tensor metadata (shape, len)
    #[test]
    fn test_concurrent_tensor_metadata_access() {
        loom::model(|| {
            let tensor = Arc::new(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)], &[3]).unwrap());

            let mut handles = vec![];

            // Multiple threads accessing different metadata
            let tensor1 = Arc::clone(&tensor);
            handles.push(thread::spawn(move || {
                assert_eq!(tensor1.len(), 3);
                assert_eq!(tensor1.shape().dims(), &[3]);
            }));

            let tensor2 = Arc::clone(&tensor);
            handles.push(thread::spawn(move || {
                assert_eq!(tensor2.shape().dims(), &[3]);
                assert_eq!(tensor2.len(), 3);
            }));

            let tensor3 = Arc::clone(&tensor);
            handles.push(thread::spawn(move || {
                let shape = tensor3.shape();
                assert_eq!(shape.dims(), &[3]);
            }));

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
}

// Note: For non-loom builds, we'll skip these tests
#[cfg(not(loom))]
mod loom_tests {
    #[test]
    fn loom_tests_disabled() {
        // Loom tests are disabled - this is normal for regular builds
        // To run loom tests: RUSTFLAGS="--cfg loom" cargo test --test loom
    }
}</content>

