//! Sequential Model Tests
//!
//! Tests for Sequential model composition and multi-layer networks.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{Linear, Module, Sequential};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

#[test]
fn test_sequential_empty() {
    let model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Empty model should have no parameters
    assert_eq!(model.parameters().len(), 0);
    assert_eq!(model.modules().len(), 0);
}

#[test]
fn test_sequential_add_modules() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Add linear layers
    model.add_module("fc1".to_string(), Linear::new(10, 5).unwrap());
    model.add_module("fc2".to_string(), Linear::new(5, 2).unwrap());

    // Should have 2 modules
    assert_eq!(model.modules().len(), 2);

    // Should have parameters from both layers: 2 layers × 2 params each = 4 total
    assert_eq!(model.parameters().len(), 4);
}

#[test]
fn test_sequential_forward_pass() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Create a 2-layer network: 4 -> 3 -> 2
    model.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    // Input: [batch_size=1, input_features=4]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4],
    )
    .unwrap();

    let output = model.forward(&input).unwrap();

    // Output should be [batch_size=1, output_features=2]
    assert_eq!(output.shape().dims(), &[1, 2]);
}

#[test]
fn test_sequential_gradient_flow() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Add layers
    model.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap()
    .requires_grad_(true);

    let output = model.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // All parameters should require gradients
    let params = model.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}

#[test]
fn test_sequential_zero_grad() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    model.add_module("fc1".to_string(), Linear::new(4, 2).unwrap());
    model.add_module("fc2".to_string(), Linear::new(2, 1).unwrap());

    // Test zero_grad functionality
    model.zero_grad();

    // Parameters should still exist
    let params = model.parameters();

    // Linear has 2 params each (weight + bias), so 2 layers × 2 = 4 params
    assert_eq!(params.len(), 4);
}

#[test]
fn test_sequential_train_eval_modes() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    model.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());

    // Test train mode
    model.train(true);

    // Test eval mode
    model.train(false);

    // Functionality should work in both modes
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap();

    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2]);
}

#[test]
fn test_sequential_module_api() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    model.add_module("fc1".to_string(), Linear::new(4, 2).unwrap());

    // Test Module trait methods
    assert_eq!(model.name(), "Sequential");

    // Test parameter access
    let params = model.parameters();
    assert_eq!(params.len(), 2); // Linear has weight + bias

    // Test module access
    let modules = model.modules();
    assert_eq!(modules.len(), 1);
}

#[test]
fn test_sequential_complex_network() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Build a more complex network
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("fc2".to_string(), Linear::new(8, 6).unwrap());
    model.add_module("fc3".to_string(), Linear::new(6, 4).unwrap());
    model.add_module("fc4".to_string(), Linear::new(4, 2).unwrap());

    // Input: [batch_size=2, input_features=10]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 10]).unwrap();

    let output = model.forward(&input).unwrap();

    // Output should be [batch_size=2, output_features=2]
    assert_eq!(output.shape().dims(), &[2, 2]);

    // Should have 4 modules and 8 parameters (4 layers × 2 params each)
    assert_eq!(model.modules().len(), 4);
    assert_eq!(model.parameters().len(), 8);
}

#[test]
fn test_sequential_different_batch_sizes() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    model.add_module("fc1".to_string(), Linear::new(4, 2).unwrap());

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8];

    for &batch_size in &batch_sizes {
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[batch_size, 4])
                .unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[batch_size, 2]);
    }
}

#[test]
fn test_sequential_module_removal() {
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    model.add_module("fc1".to_string(), Linear::new(4, 2).unwrap());
    model.add_module("fc2".to_string(), Linear::new(2, 1).unwrap());

    assert_eq!(model.modules().len(), 2);
    assert_eq!(model.parameters().len(), 4);

    // Note: Sequential doesn't currently have removal API
    // This test documents current behavior
}

#[test]
fn test_sequential_gradient_consistency() {
    let mut model1 = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
    let mut model2 = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Same architecture
    model1.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());
    model2.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap();

    let output1 = model1.forward(&input).unwrap();
    let output2 = model2.forward(&input).unwrap();

    // Outputs should have same shape (values may differ due to random init)
    assert_eq!(output1.shape(), output2.shape());

    // Both models should have same number of parameters
    assert_eq!(model1.parameters().len(), model2.parameters().len());
}
