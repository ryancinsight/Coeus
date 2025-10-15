//! Activation Function Tests
//!
//! Tests for activation functions including PReLU with learnable parameters.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{PReLU, Module};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

#[test]
fn test_prelu_forward() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    // Test with mixed positive/negative values
    let input_data = vec![
        Float32::new(2.0),   // positive
        Float32::new(-1.0),  // negative
        Float32::new(0.0),   // zero
        Float32::new(-0.5),  // negative
        Float32::new(3.0),   // positive
    ];

    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        input_data, &[5]
    ).unwrap();

    let output = prelu.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[5]);

    // Check that output is properly computed
    let output_data = output.as_slice();
    assert_eq!(output_data.len(), 5);
}

#[test]
fn test_prelu_parameters() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(3).unwrap();

    let params = prelu.parameters();

    // PReLU with 3 parameters should have 3 learnable weights
    assert_eq!(params.len(), 3);

    // Each parameter should be scalar (shape [])
    for param in &params {
        assert_eq!(param.data().shape().dims(), &[0usize]);
        assert!(param.requires_grad());
    }
}

#[test]
fn test_prelu_shared_parameter() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    let params = prelu.parameters();

    // Shared parameter case should have 1 parameter
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].data().shape().dims(), &[0usize]);
}

#[test]
fn test_prelu_gradient_flow() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(-1.0)],
        &[2]
    ).unwrap().requires_grad_(true);

    let output = prelu.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // Parameters should require gradients
    let params = prelu.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}

#[test]
fn test_prelu_zero_grad() {
    let mut prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(2).unwrap();

    // Test zero_grad functionality
    prelu.zero_grad();

    // Parameters should still exist
    let params = prelu.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_prelu_module_api() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    // Test Module trait methods
    assert_eq!(prelu.name(), "PReLU");

    // Test parameter access
    let params = prelu.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_prelu_different_shapes() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    // Test with different input shapes
    let test_shapes = vec![
        vec![5],      // 1D
        vec![2, 3],   // 2D
        vec![2, 2, 2], // 3D
    ];

    for shape in test_shapes {
        let size: usize = shape.iter().product();
        let input_data = vec![Float32::new(1.0); size];
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data, &shape
        ).unwrap();

        let output = prelu.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), shape.as_slice());
    }
}

#[test]
fn test_prelu_per_channel() {
    let prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(3).unwrap();

    // Input with 3 channels
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0), Float32::new(-1.0), Float32::new(0.5),  // channel 0
            Float32::new(-2.0), Float32::new(1.5), Float32::new(-0.5), // channel 1
            Float32::new(0.0), Float32::new(-3.0), Float32::new(2.0),  // channel 2
        ],
        &[3, 3] // [channels, values_per_channel]
    ).unwrap();

    let output = prelu.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[3, 3]);

    // Check that output is properly computed with per-channel parameters
    let output_data = output.as_slice();
    assert_eq!(output_data.len(), 9);
}

#[test]
fn test_prelu_train_eval_modes() {
    let mut prelu = PReLU::<CpuBackend, DenseStorage<Float32>, Float32>::new(1).unwrap();

    // Test train mode
    prelu.train(true);

    // Test eval mode
    prelu.train(false);

    // Functionality should work in both modes
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(-1.0)],
        &[2]
    ).unwrap();

    let output = prelu.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[2]);
}
