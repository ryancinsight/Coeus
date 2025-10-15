//! Integration tests for neural network components.
//!
//! These tests verify that neural network modules work together
//! to build and train complete models.

use super::*;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

/// Test a simple feedforward neural network.
#[test]
fn test_simple_feedforward_network() -> Result<()> {
    // Create a simple 2-layer neural network: Linear(10, 5) -> ReLU -> Linear(5, 1)
    let mut model = Sequential::new();

    // Add layers
    model.add_module("linear1".to_string(), Linear::new(10, 5)?);
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("linear2".to_string(), Linear::new(5, 1)?);

    // Create input tensor (batch_size=2, input_dim=10)
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            // Batch 1
            Float32::new(1.0), Float32::new(0.5), Float32::new(-0.2), Float32::new(0.8),
            Float32::new(-1.5), Float32::new(0.3), Float32::new(1.2), Float32::new(-0.7),
            Float32::new(0.9), Float32::new(-0.4),
            // Batch 2
            Float32::new(0.2), Float32::new(-0.8), Float32::new(1.1), Float32::new(-0.3),
            Float32::new(0.6), Float32::new(-1.2), Float32::new(0.4), Float32::new(0.9),
            Float32::new(-0.5), Float32::new(1.3),
        ],
        &[2, 10],
    )?;

    // Forward pass
    let output = model.forward(&input)?;

    // Check output shape
    assert_eq!(output.shape().dims(), &[2, 1], "Output should be [batch_size, 1]");

    // Check that output is finite (not NaN or infinite)
    let output_slice = output.as_slice();
    for &val in output_slice {
        assert!(val.get().is_finite(), "Output should be finite");
    }

    Ok(())
}

/// Test model parameter management and state dict operations.
#[test]
fn test_model_parameters_and_state_dict() -> Result<()> {
    // Create a simple model
    let mut model = Sequential::new();
    model.add_module("linear".to_string(), Linear::new(4, 2)?);
    model.add_module("relu".to_string(), ReLU::new());

    // Check that model has parameters
    let params = model.parameters();
    assert!(!params.is_empty(), "Model should have parameters");

    // Check parameter names and shapes
    let expected_shapes = vec![vec![4, 2], vec![2]]; // weight and bias for Linear layer
    let mut param_idx = 0;

    for param in &params {
        if param_idx < expected_shapes.len() {
            assert_eq!(param.data().shape().dims(), expected_shapes[param_idx],
                      "Parameter {} should have correct shape", param_idx);
        }
        param_idx += 1;
    }

    // Test state dict save/load
    let state_dict = model.state_dict();

    // Create a new model and load state
    let mut new_model = Sequential::new();
    new_model.add_module("linear".to_string(), Linear::new(4, 2)?);
    new_model.add_module("relu".to_string(), ReLU::new());

    new_model.load_state_dict(&state_dict);

    // Parameters should be identical
    let new_params = new_model.parameters();
    assert_eq!(params.len(), new_params.len(), "Models should have same number of parameters");

    for (old_param, new_param) in params.iter().zip(new_params.iter()) {
        let old_data = old_param.data().as_slice();
        let new_data = new_param.data().as_slice();
        assert_eq!(old_data, new_data, "Parameters should be identical after load");
    }

    Ok(())
}

/// Test loss function integration.
#[test]
fn test_loss_function_integration() -> Result<()> {
    // Create some dummy predictions and targets
    let predictions = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(0.8), Float32::new(0.2),  // Sample 1: high confidence for class 0
            Float32::new(0.3), Float32::new(0.7),  // Sample 2: high confidence for class 1
        ],
        &[2, 2],
    )?;

    let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0), Float32::new(0.0),  // Sample 1: true class 0
            Float32::new(0.0), Float32::new(1.0),  // Sample 2: true class 1
        ],
        &[2, 2],
    )?;

    // Test CrossEntropyLoss using functional API
    let loss = functional::cross_entropy(&predictions, &targets)?;

    // Check that loss is positive and finite
    let loss_slice = loss.as_slice();
    for &val in loss_slice {
        assert!(val.get() > 0.0, "Cross entropy loss should be positive");
        assert!(val.get().is_finite(), "Loss should be finite");
    }

    // Test MSE loss using functional API
    let mse_loss = functional::mse_loss(&predictions, &targets)?;

    let mse_slice = mse_loss.as_slice();
    for &val in mse_slice {
        assert!(val.get() >= 0.0, "MSE loss should be non-negative");
        assert!(val.get().is_finite(), "MSE loss should be finite");
    }

    Ok(())
}

/// Test activation functions.
#[test]
fn test_activation_functions() -> Result<()> {
    // Test ReLU
    let relu = ReLU::new();
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0), Float32::new(-0.5), Float32::new(0.0), Float32::new(2.0),
        ],
        &[4],
    )?;

    let relu_output = relu.forward(&input)?;
    let relu_slice = relu_output.as_slice();

    assert_eq!(relu_slice[0].get(), 1.0, "Positive values should pass through");
    assert_eq!(relu_slice[1].get(), 0.0, "Negative values should be clamped to 0");
    assert_eq!(relu_slice[2].get(), 0.0, "Zero should remain zero");
    assert_eq!(relu_slice[3].get(), 2.0, "Positive values should pass through");

    // Test Sigmoid
    let sigmoid = Sigmoid::new();
    let sigmoid_output = sigmoid.forward(&input)?;
    let sigmoid_slice = sigmoid_output.as_slice();

    // Check that sigmoid outputs are in (0, 1) range
    for &val in sigmoid_slice {
        let v = val.get();
        assert!(v > 0.0 && v < 1.0, "Sigmoid should output values in (0, 1)");
    }

    Ok(())
}

/// Test functional API.
#[test]
fn test_functional_api() -> Result<()> {
    // Create test tensors
    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0), Float32::new(2.0),
            Float32::new(3.0), Float32::new(4.0),
        ],
        &[2, 2],
    )?;

    // Test functional linear transformation
    let weight = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0), Float32::new(0.0),
            Float32::new(0.0), Float32::new(1.0),
        ],
        &[2, 2],
    )?;

    let bias = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(0.1), Float32::new(0.2)],
        &[2],
    )?;

    let output = functional::linear(&input, &weight, Some(&bias))?;

    // Check output shape
    assert_eq!(output.shape().dims(), &[2, 2], "Linear output should have correct shape");

    // Test functional ReLU
    let relu_output = functional::relu(&output)?;
    let relu_slice = relu_output.as_slice();

    // All values should be non-negative
    for &val in relu_slice {
        assert!(val.get() >= 0.0, "ReLU should output non-negative values");
    }

    Ok(())
}
