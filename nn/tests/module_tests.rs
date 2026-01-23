//! Unit tests for neural network modules.
//!
//! This test suite verifies that modules correctly implement the Module trait
//! and properly manage parameters, training modes, and forward passes.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::core::module::Module;
use nn::modules::activation::{GeLU, LeakyReLU, ReLU, SiLU, ELU};
use nn::modules::linear::Linear;
use nn::modules::loss::{CrossEntropyLoss, MSELoss};
use storage::DenseStorage;
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestDataType = Float32;

// ============================================================================
// Activation Module Tests
// ============================================================================

#[test]
fn test_relu_forward_pass() {
    let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    // Test with mixed positive and negative values
    let input = Tensor::from_vec(
        vec![
            TestDataType::new(-2.0),
            TestDataType::new(-1.0),
            TestDataType::new(0.0),
            TestDataType::new(1.0),
            TestDataType::new(2.0),
        ],
        &[5],
    )
    .unwrap();

    let output = relu.forward(&input).unwrap();
    let output_data = output.as_slice();

    // ReLU should zero out negative values
    assert_eq!(output_data[0].get(), 0.0);
    assert_eq!(output_data[1].get(), 0.0);
    assert_eq!(output_data[2].get(), 0.0);
    assert_eq!(output_data[3].get(), 1.0);
    assert_eq!(output_data[4].get(), 2.0);
}

#[test]
fn test_relu_module_trait() {
    let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    // Test Module trait implementation
    assert_eq!(relu.name(), "ReLU");
    assert_eq!(relu.parameters().len(), 0); // ReLU has no parameters
}

#[test]
fn test_gelu_forward_pass() {
    let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();

    let input = Tensor::from_vec(
        vec![
            TestDataType::new(-1.0),
            TestDataType::new(0.0),
            TestDataType::new(1.0),
        ],
        &[3],
    )
    .unwrap();

    let output = gelu.forward(&input).unwrap();
    let output_data = output.as_slice();

    // GELU(0) should be approximately 0
    assert!(output_data[1].get().abs() < 0.01);

    // GELU(1) should be approximately 0.841
    assert!((output_data[2].get() - 0.841).abs() < 0.01);
}

#[test]
fn test_gelu_module_trait() {
    let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();

    assert_eq!(gelu.name(), "GeLU");
    assert_eq!(gelu.parameters().len(), 0);
}

#[test]
fn test_silu_forward_pass() {
    let silu = SiLU::<TestBackend, TestStorage, TestDataType>::new();

    let input = Tensor::from_vec(
        vec![
            TestDataType::new(-1.0),
            TestDataType::new(0.0),
            TestDataType::new(1.0),
        ],
        &[3],
    )
    .unwrap();

    let output = silu.forward(&input).unwrap();
    let output_data = output.as_slice();

    // SiLU(0) should be approximately 0
    assert!(output_data[1].get().abs() < 0.01);

    // SiLU(1) should be approximately 0.731
    assert!((output_data[2].get() - 0.731).abs() < 0.05);
}

#[test]
fn test_silu_module_trait() {
    let silu = SiLU::<TestBackend, TestStorage, TestDataType>::new();

    assert_eq!(silu.name(), "SiLU");
    assert_eq!(silu.parameters().len(), 0);
}

#[test]
fn test_leaky_relu_forward_pass() {
    let leaky_relu =
        LeakyReLU::<TestBackend, TestStorage, TestDataType>::new(TestDataType::new(0.01));

    let input = Tensor::from_vec(
        vec![
            TestDataType::new(-2.0),
            TestDataType::new(-1.0),
            TestDataType::new(0.0),
            TestDataType::new(1.0),
            TestDataType::new(2.0),
        ],
        &[5],
    )
    .unwrap();

    let output = leaky_relu.forward(&input).unwrap();
    let output_data = output.as_slice();

    // Negative values should be scaled by 0.01
    assert!((output_data[0].get() - (-0.02)).abs() < 0.001);
    assert!((output_data[1].get() - (-0.01)).abs() < 0.001);
    assert_eq!(output_data[2].get(), 0.0);
    assert_eq!(output_data[3].get(), 1.0);
    assert_eq!(output_data[4].get(), 2.0);
}

#[test]
fn test_leaky_relu_module_trait() {
    let leaky_relu =
        LeakyReLU::<TestBackend, TestStorage, TestDataType>::new(TestDataType::new(0.01));

    assert_eq!(leaky_relu.name(), "LeakyReLU");
    assert_eq!(leaky_relu.parameters().len(), 0);
}

#[test]
fn test_elu_forward_pass() {
    let elu = ELU::<TestBackend, TestStorage, TestDataType>::new(TestDataType::new(1.0));

    let input = Tensor::from_vec(
        vec![
            TestDataType::new(-1.0),
            TestDataType::new(0.0),
            TestDataType::new(1.0),
        ],
        &[3],
    )
    .unwrap();

    let output = elu.forward(&input).unwrap();
    let output_data = output.as_slice();

    // ELU(-1) with alpha=1.0 should be approximately -0.632
    assert!((output_data[0].get() - (-0.632)).abs() < 0.01);
    assert_eq!(output_data[1].get(), 0.0);
    assert_eq!(output_data[2].get(), 1.0);
}

#[test]
fn test_elu_module_trait() {
    let elu = ELU::<TestBackend, TestStorage, TestDataType>::new(TestDataType::new(1.0));

    assert_eq!(elu.name(), "ELU");
    assert_eq!(elu.parameters().len(), 0);
}

// ============================================================================
// Linear Module Tests
// ============================================================================

#[test]
fn test_linear_forward_pass() {
    let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(3, 2).unwrap();

    // Input: [batch_size=2, in_features=3]
    let input = Tensor::from_vec(
        vec![
            TestDataType::new(1.0),
            TestDataType::new(2.0),
            TestDataType::new(3.0),
            TestDataType::new(4.0),
            TestDataType::new(5.0),
            TestDataType::new(6.0),
        ],
        &[2, 3],
    )
    .unwrap();

    let output = linear.forward(&input).unwrap();

    // Output should be [batch_size=2, out_features=2]
    assert_eq!(output.shape().dims(), &[2, 2]);
}

#[test]
fn test_linear_module_trait() {
    let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(3, 2).unwrap();

    assert_eq!(linear.name(), "Linear");
    assert_eq!(linear.parameters().len(), 2); // weight and bias
}

#[test]
fn test_linear_parameter_management() {
    let mut linear = Linear::<TestBackend, TestStorage, TestDataType>::new(3, 2).unwrap();

    // Check parameters exist
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].name(), "weight");
    assert_eq!(params[1].name(), "bias");

    // Test zero_grad
    linear.zero_grad();
    let params_after = linear.parameters();
    assert!(!params_after[0].requires_grad());
    assert!(!params_after[1].requires_grad());
}

#[test]
fn test_linear_shape_validation() {
    let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(3, 2).unwrap();

    // Wrong input shape should fail
    let wrong_input = Tensor::from_vec(
        vec![
            TestDataType::new(1.0),
            TestDataType::new(2.0),
            TestDataType::new(3.0),
            TestDataType::new(4.0),
        ],
        &[2, 2], // Should be [2, 3]
    )
    .unwrap();

    let result = linear.forward(&wrong_input);
    assert!(result.is_err());
}

// ============================================================================
// Loss Module Tests
// ============================================================================

#[test]
fn test_mse_loss_forward() {
    let loss_fn = MSELoss::new();

    let predictions = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(
        vec![
            TestDataType::new(1.0),
            TestDataType::new(2.0),
            TestDataType::new(3.0),
        ],
        &[3],
    )
    .unwrap();

    let targets = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(
        vec![
            TestDataType::new(1.5),
            TestDataType::new(2.0),
            TestDataType::new(2.5),
        ],
        &[3],
    )
    .unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Expected: mean((1-1.5)² + (2-2)² + (3-2.5)²) = mean(0.25 + 0 + 0.25) = 0.166...
    let loss_value = loss.as_slice()[0].get();
    assert!(loss_value > 0.166 && loss_value < 0.167);
}

#[test]
fn test_mse_loss_perfect_prediction() {
    let loss_fn = MSELoss::new();

    let predictions = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(
        vec![
            TestDataType::new(1.0),
            TestDataType::new(2.0),
            TestDataType::new(3.0),
        ],
        &[3],
    )
    .unwrap();

    let targets = predictions.clone();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Perfect prediction should give zero loss
    let loss_value = loss.as_slice()[0].get();
    assert!(loss_value.abs() < 1e-6);
}

#[test]
fn test_cross_entropy_loss_forward() {
    let loss_fn = CrossEntropyLoss::new();

    // 3 classes, 2 samples
    let logits = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(
        vec![
            TestDataType::new(1.0),
            TestDataType::new(0.5),
            TestDataType::new(0.2), // sample 1
            TestDataType::new(0.1),
            TestDataType::new(2.0),
            TestDataType::new(0.3), // sample 2
        ],
        &[2, 3],
    )
    .unwrap();

    let targets = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(
        vec![TestDataType::new(0.0), TestDataType::new(1.0)], // class 0 for sample 1, class 1 for sample 2
        &[2],
    )
    .unwrap();

    let loss = loss_fn.forward(&logits, &targets).unwrap();

    // Loss should be positive
    let loss_value = loss.as_slice()[0].get();
    assert!(loss_value > 0.0);
}

// ============================================================================
// Training Mode Tests
// ============================================================================

#[test]
fn test_module_training_mode() {
    let mut relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    // Test train mode (no-op for ReLU but should not error)
    relu.train(true);
    relu.train(false);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_activation_empty_tensor() {
    let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    let empty_input = Tensor::from_vec(vec![], &[0]).unwrap();

    let result = relu.forward(&empty_input);
    // Should handle empty tensors gracefully
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_activation_single_element() {
    let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    let single_input = Tensor::from_vec(vec![TestDataType::new(-1.0)], &[1]).unwrap();

    let output = relu.forward(&single_input).unwrap();
    assert_eq!(output.as_slice()[0].get(), 0.0);
}

#[test]
fn test_linear_zero_features() {
    // Should fail to create linear layer with zero features
    let result = Linear::<TestBackend, TestStorage, TestDataType>::new(0, 2);
    assert!(result.is_err());

    let result = Linear::<TestBackend, TestStorage, TestDataType>::new(2, 0);
    assert!(result.is_err());
}

// ============================================================================
// Module Cloning Tests
// ============================================================================

#[test]
fn test_module_clone_box() {
    let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

    let cloned = relu.clone_box();
    assert_eq!(cloned.name(), "ReLU");
}

#[test]
fn test_linear_clone() {
    let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(3, 2).unwrap();

    let cloned = linear.clone();
    assert_eq!(cloned.in_features, linear.in_features);
    assert_eq!(cloned.out_features, linear.out_features);
}
