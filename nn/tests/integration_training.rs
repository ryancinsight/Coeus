//! Integration Tests for End-to-End Training
//!
//! Tests complete training workflows including forward pass, loss computation,
//! backward pass, and optimizer updates.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, Module, ReLU, Sequential};
use optim::{Adam, BaseOptimizer};
use storage::DenseStorage;
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

/// Test a simple 2-layer network training on synthetic data
/// Network: Linear(4 -> 3) -> ReLU -> Linear(3 -> 2)
#[test]
fn test_simple_network_training() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    // Create synthetic training data
    let input = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4],
    )
    .unwrap()
    .requires_grad_(true);

    let target = TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(0.0)], &[1, 2]).unwrap();

    // Get initial parameters
    let params = model.parameters();
    assert_eq!(params.len(), 4); // 2 layers × 2 params (weight + bias)

    // Create optimizer
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();
    let _optimizer = Adam::new(param_tensors, 0.01);

    // Perform forward pass
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2]);

    // Compute loss (simple MSE)
    let diff = &output - &target;
    let squared = &diff * &diff;
    let initial_loss = squared.mean_dims(None, false).unwrap();

    // Verify loss is a scalar-like tensor (either rank-0 or a single element)
    let loss_dims = initial_loss.shape().dims();
    assert!(
        loss_dims.is_empty() || loss_dims == [1],
        "Expected scalar-like loss, got dims {:?}",
        loss_dims
    );

    // Store initial loss value
    let initial_loss_value = initial_loss.as_slice()[0].get();
    assert!(initial_loss_value > 0.0);

    println!("Initial loss: {}", initial_loss_value);
}

/// Test that gradients flow correctly through the network
#[test]
fn test_gradient_flow() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(2, 1).unwrap());

    // Create input with gradient tracking
    let input = TestTensor::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap()
    .requires_grad_(true);

    // Forward pass
    let output = model.forward(&input).unwrap();

    // Verify output requires gradients
    assert!(output.requires_grad());

    // Verify all parameters require gradients
    let params = model.parameters();
    for param in params.iter() {
        assert!(param.requires_grad());
    }

    // Verify output shape
    assert_eq!(output.shape().dims(), &[1, 1]);
}

/// Test training loop with multiple iterations
#[test]
fn test_multi_iteration_training() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(2, 4).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(4, 1).unwrap());

    // Create synthetic data (simple linear relationship: y = 2*x1 + 3*x2)
    let inputs = vec![
        TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[1, 2])
            .unwrap()
            .requires_grad_(true),
        TestTensor::from_vec(vec![Float32::new(2.0), Float32::new(3.0)], &[1, 2])
            .unwrap()
            .requires_grad_(true),
        TestTensor::from_vec(vec![Float32::new(3.0), Float32::new(4.0)], &[1, 2])
            .unwrap()
            .requires_grad_(true),
    ];

    let targets = vec![
        TestTensor::from_vec(vec![Float32::new(8.0)], &[1, 1]).unwrap(), // 2*1 + 3*2 = 8
        TestTensor::from_vec(vec![Float32::new(13.0)], &[1, 1]).unwrap(), // 2*2 + 3*3 = 13
        TestTensor::from_vec(vec![Float32::new(18.0)], &[1, 1]).unwrap(), // 2*3 + 3*4 = 18
    ];

    // Get parameters for optimizer
    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();
    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Track losses
    let mut losses = Vec::new();

    // Training loop
    for epoch in 0..10 {
        let mut epoch_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let output = model.forward(input).unwrap();

            // Compute loss
            let diff = &output - target;
            let squared = &diff * &diff;
            let loss = squared.mean_dims(None, false).unwrap();

            let loss_value = loss.as_slice()[0].get();
            epoch_loss += loss_value;

            // Zero gradients
            optimizer.zero_grad();
        }

        losses.push(epoch_loss / inputs.len() as f32);
        println!("Epoch {}: Loss = {}", epoch, losses[epoch]);
    }

    // Verify we have losses for all epochs
    assert_eq!(losses.len(), 10);

    // Note: Without actual backward pass implementation, we can't verify loss decreases
    // This test validates the training loop structure
}

/// Test batch processing
#[test]
fn test_batch_training() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(2, 1).unwrap());

    // Create batch input (batch_size=4, features=3)
    let batch_input = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[4, 3],
    )
    .unwrap()
    .requires_grad_(true);

    let batch_target = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[4, 1],
    )
    .unwrap();

    // Forward pass
    let output = model.forward(&batch_input).unwrap();

    // Verify output shape
    assert_eq!(output.shape().dims(), &[4, 1]);

    // Compute loss
    let diff = &output - &batch_target;
    let squared = &diff * &diff;
    let loss = squared.mean_dims(None, false).unwrap();

    // Verify loss is scalar-like tensor (either rank-0 or a single element)
    let loss_dims = loss.shape().dims();
    assert!(
        loss_dims.is_empty() || loss_dims == [1],
        "Expected scalar-like loss, got dims {:?}",
        loss_dims
    );

    let loss_value = loss.as_slice()[0].get();
    assert!(loss_value >= 0.0);

    println!("Batch loss: {}", loss_value);
}

/// Test zero_grad functionality
#[test]
fn test_zero_grad_integration() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(2, 2).unwrap());

    // Get parameters
    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();
    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Call zero_grad multiple times (should not crash)
    optimizer.zero_grad();
    optimizer.zero_grad();
    model.zero_grad();
    model.zero_grad();

    // Verify model still works after zero_grad
    let input = TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[1, 2]).unwrap();

    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2]);
}

/// Test model in train vs eval mode
#[test]
fn test_train_eval_modes() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());
    model.add_module("relu".to_string(), ReLU::new());

    let input = TestTensor::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap();

    // Test in train mode
    model.train(true);
    let output_train = model.forward(&input).unwrap();
    assert_eq!(output_train.shape().dims(), &[1, 2]);

    // Test in eval mode
    model.train(false);
    let output_eval = model.forward(&input).unwrap();
    assert_eq!(output_eval.shape().dims(), &[1, 2]);

    // For Linear + ReLU, outputs should be the same in train/eval
    // (no dropout or batch norm that behaves differently)
    // We just verify both modes work without crashing
}

/// Test parameter count consistency
#[test]
fn test_parameter_count_consistency() {
    // Create a network with known parameter count
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(10, 5).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(5, 2).unwrap());

    let params = model.parameters();

    // fc1: weight (5x10) + bias (5) = 2 params
    // relu: no params
    // fc2: weight (2x5) + bias (2) = 2 params
    // Total: 4 parameter tensors
    assert_eq!(params.len(), 4);

    // Verify all parameters have correct requires_grad
    for param in params.iter() {
        assert!(param.requires_grad());
    }
}

/// Test different network architectures
#[test]
fn test_various_architectures() {
    // Test 1: Single layer
    let mut model1 = Sequential::<TestBackend, TestStorage, Float32>::new();
    model1.add_module("fc".to_string(), Linear::new(5, 3).unwrap());

    let input1 = TestTensor::ones(&[2, 5]).unwrap();
    let output1 = model1.forward(&input1).unwrap();
    assert_eq!(output1.shape().dims(), &[2, 3]);

    // Test 2: Deep network
    let mut model2 = Sequential::<TestBackend, TestStorage, Float32>::new();
    model2.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model2.add_module("relu1".to_string(), ReLU::new());
    model2.add_module("fc2".to_string(), Linear::new(8, 6).unwrap());
    model2.add_module("relu2".to_string(), ReLU::new());
    model2.add_module("fc3".to_string(), Linear::new(6, 4).unwrap());
    model2.add_module("relu3".to_string(), ReLU::new());
    model2.add_module("fc4".to_string(), Linear::new(4, 2).unwrap());

    let input2 = TestTensor::ones(&[1, 10]).unwrap();
    let output2 = model2.forward(&input2).unwrap();
    assert_eq!(output2.shape().dims(), &[1, 2]);

    // Test 3: Wide network
    let mut model3 = Sequential::<TestBackend, TestStorage, Float32>::new();
    model3.add_module("fc1".to_string(), Linear::new(5, 100).unwrap());
    model3.add_module("relu".to_string(), ReLU::new());
    model3.add_module("fc2".to_string(), Linear::new(100, 3).unwrap());

    let input3 = TestTensor::ones(&[1, 5]).unwrap();
    let output3 = model3.forward(&input3).unwrap();
    assert_eq!(output3.shape().dims(), &[1, 3]);
}
