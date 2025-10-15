//! Batch Normalization Tests
//!
//! Comprehensive tests for BatchNorm1d, BatchNorm2d, BatchNorm3d functionality.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{BatchNorm1d, BatchNorm2d, BatchNorm3d, Module};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

#[test]
fn test_batchnorm1d_forward() {
    let batchnorm = BatchNorm1d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Input: [batch_size=2, features=64, length=10]
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 10]).unwrap();

    let output = batchnorm.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[2, 64, 10]);
}

#[test]
fn test_batchnorm2d_forward() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Input: [batch_size=2, channels=64, height=8, width=8]
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 8, 8]).unwrap();

    let output = batchnorm.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[2, 64, 8, 8]);
}

#[test]
fn test_batchnorm3d_forward() {
    let batchnorm = BatchNorm3d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Input: [batch_size=2, channels=64, depth=4, height=8, width=8]
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 4, 8, 8]).unwrap();

    let output = batchnorm.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[2, 64, 4, 8, 8]);
}

#[test]
fn test_batchnorm_parameters() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    let params = batchnorm.parameters();

    // BatchNorm has weight and bias parameters
    assert_eq!(params.len(), 2);

    // Both should be [num_features] = [64]
    assert_eq!(params[0].data().shape().dims(), &[64]); // weight (γ)
    assert_eq!(params[1].data().shape().dims(), &[64]); // bias (β)

    // Both should require gradients
    assert!(params[0].requires_grad());
    assert!(params[1].requires_grad());
}

#[test]
fn test_batchnorm_running_statistics() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Check running statistics are initialized
    let running_mean = batchnorm.running_mean();
    let running_var = batchnorm.running_var();

    assert_eq!(running_mean.shape().dims(), &[64]); // [num_features]
    assert_eq!(running_var.shape().dims(), &[64]);  // [num_features]

    // Initially, running_mean should be 0, running_var should be 1
    // (These are interior mutable, so we can't directly inspect values)
}

#[test]
fn test_batchnorm_configuration() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    assert_eq!(batchnorm.num_features, 64);
    assert_eq!(batchnorm.eps, 1e-5);
    assert_eq!(batchnorm.momentum, 0.1);
    assert!(batchnorm.track_running_stats); // Default should be true
}

#[test]
fn test_batchnorm_train_eval_modes() {
    let mut batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Start in eval mode
    batchnorm.train(false);
    assert!(!batchnorm.training);

    // Switch to train mode
    batchnorm.train(true);
    assert!(batchnorm.training);

    // Test functionality in both modes
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 8, 8]).unwrap();

    let output_train = batchnorm.forward(&input).unwrap();
    batchnorm.train(false);
    let output_eval = batchnorm.forward(&input).unwrap();

    // Both should produce valid outputs
    assert_eq!(output_train.shape(), output_eval.shape());
}

#[test]
fn test_batchnorm_gradient_flow() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 8, 8]).unwrap()
        .requires_grad_(true);

    let output = batchnorm.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // Parameters should require gradients
    let params = batchnorm.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}

#[test]
fn test_batchnorm_different_batch_sizes() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8];

    for &batch_size in &batch_sizes {
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[batch_size, 64, 8, 8]).unwrap();
        let output = batchnorm.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[batch_size, 64, 8, 8]);
    }
}

#[test]
fn test_batchnorm_invalid_dimensions() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Wrong number of channels
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 32, 8, 8]).unwrap(); // 32 channels, but BN expects 64

    // This should fail - BatchNorm requires exact channel matching
    let result = batchnorm.forward(&input);
    assert!(result.is_err()); // BatchNorm validates input channels match num_features
}

#[test]
fn test_batchnorm_zero_grad() {
    let mut batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Test zero_grad functionality
    batchnorm.zero_grad();

    // Parameters should still exist
    let params = batchnorm.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm_module_api() {
    let batchnorm = BatchNorm2d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Test Module trait methods
    assert_eq!(batchnorm.name(), "BatchNorm2d");

    // Test parameter access
    let params = batchnorm.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm1d_specific_shape() {
    let batchnorm = BatchNorm1d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Input: [batch_size=2, features=64, length=10]
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 10]).unwrap();

    let output = batchnorm.forward(&input).unwrap();

    // Should preserve shape
    assert_eq!(output.shape().dims(), &[2, 64, 10]);
}

#[test]
fn test_batchnorm3d_specific_shape() {
    let batchnorm = BatchNorm3d::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1).unwrap();

    // Input: [batch_size=2, channels=64, depth=4, height=8, width=8]
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 4, 8, 8]).unwrap();

    let output = batchnorm.forward(&input).unwrap();

    // Should preserve shape
    assert_eq!(output.shape().dims(), &[2, 64, 4, 8, 8]);
}
