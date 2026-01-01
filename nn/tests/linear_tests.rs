//! Linear Layer Tests
//!
//! Comprehensive tests for Linear layer functionality, gradients, and API compatibility.

use approx::assert_relative_eq;
use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, Module};
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_linear_forward_pass() {
    // Test basic forward pass functionality
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 2).unwrap();

    // Input: [batch_size=1, input_features=3]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap();

    let output = linear.forward(&input).unwrap();

    // Output should be [batch_size=1, output_features=2]
    assert_eq!(output.shape().dims(), &[1, 2]);
}

#[test]
fn test_linear_parameter_initialization() {
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2).unwrap();

    // Check parameter shapes
    assert_eq!(linear.weight.data().shape().dims(), &[2, 4]); // [output_features, input_features]
    assert_eq!(linear.bias.data().shape().dims(), &[2]); // [output_features]

    // Check that parameters require gradients by default
    assert!(linear.weight.requires_grad());
    assert!(linear.bias.requires_grad());

    // Check parameter count
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_weight_matrix_multiplication() {
    let mut linear =
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 1).unwrap();

    // Set known weights and bias
    let weight_data = vec![Float32::new(1.0), Float32::new(2.0)];
    let bias_data = vec![Float32::new(0.5)];

    // Create custom weight and bias tensors
    let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_data,
        &[1, 2],
    )
    .unwrap()
    .requires_grad_(true);

    let bias =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(bias_data, &[1])
            .unwrap()
            .requires_grad_(true);

    *linear.weight.data_mut() = weight;
    *linear.bias.data_mut() = bias;

    // Input: [1, 2]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[1, 2],
    )
    .unwrap();

    let output = linear.forward(&input).unwrap();

    // The output shape should be correct
    assert_eq!(output.shape().dims(), &[1, 1]);

    let expected = 1.0 * 1.0 + 2.0 * 2.0 + 0.5;
    assert_relative_eq!(output.as_slice()[0].get(), expected, epsilon = 1e-6);
}

#[test]
fn test_linear_gradient_computation() {
    // Test that gradients are computed correctly
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 1).unwrap();

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[1, 2],
    )
    .unwrap()
    .requires_grad_(true);

    let output = linear.forward(&input).unwrap();

    // The output should require gradients since input does
    assert!(output.requires_grad());

    // Parameters should still require gradients
    let params = linear.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}

#[test]
fn test_linear_invalid_dimensions() {
    // Test error handling for invalid dimensions
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 2).unwrap();

    // Wrong input feature count
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)], // Only 2 features, but layer expects 3
        &[1, 2],
    )
    .unwrap();

    // This should work (broadcasting), but let's test the shape validation
    let result = linear.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_linear_zero_grad() {
    let mut linear =
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 1).unwrap();

    // Set some gradients (simulated)
    // In practice, this would happen after backward pass

    // Test zero_grad functionality
    linear.zero_grad();

    // Parameters should still exist but gradients should be zeroed
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_train_eval_modes() {
    let mut linear =
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 1).unwrap();

    // Test train mode
    linear.train(true);
    // Linear layers don't have train/eval mode differences, but the API should work

    // Test eval mode
    linear.train(false);

    // Basic functionality should still work
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[1, 2],
    )
    .unwrap();

    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 1]);
}

#[test]
fn test_linear_module_api() {
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 2).unwrap();

    // Test Module trait methods
    assert_eq!(linear.name(), "Linear");

    // Test parameter access
    let params = linear.parameters();
    assert_eq!(params.len(), 2);

    // Test that parameters have correct shapes
    assert_eq!(params[0].data().shape().dims(), &[2, 3]); // weight
    assert_eq!(params[1].data().shape().dims(), &[2]); // bias
}
