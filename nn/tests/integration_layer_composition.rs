//! Integration Tests for Layer Composition
//!
//! Tests for Sequential container, nested modules, and parameter collection.
//! Validates Requirements 15.2

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{GeLU, Linear, Module, ReLU, Sequential};
use storage::DenseStorage;
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

/// Test Sequential container with multiple layers
#[test]
fn test_sequential_container_basic() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();

    // Add multiple layers
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("relu1".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(8, 6).unwrap());
    model.add_module("relu2".to_string(), ReLU::new());
    model.add_module("fc3".to_string(), Linear::new(6, 4).unwrap());

    // Verify module count
    assert_eq!(model.modules().len(), 5);

    // Verify parameter count (3 Linear layers × 2 params each = 6 params)
    let params = model.parameters();
    assert_eq!(params.len(), 6);

    // Test forward pass
    let input = TestTensor::ones(&[2, 10]).unwrap();
    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape().dims(), &[2, 4]);
}

/// Test Sequential container with different activation functions
#[test]
fn test_sequential_mixed_activations() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();

    // Mix different activation functions
    model.add_module("fc1".to_string(), Linear::new(5, 4).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("gelu".to_string(), GeLU::new());
    model.add_module("fc3".to_string(), Linear::new(3, 2).unwrap());

    let input = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ],
        &[1, 5],
    )
    .unwrap();

    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2]);
}

/// Test nested Sequential containers
#[test]
fn test_nested_sequential_containers() {
    // Create inner sequential block
    let mut inner_block = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner_block.add_module("fc1".to_string(), Linear::new(8, 6).unwrap());
    inner_block.add_module("relu".to_string(), ReLU::new());
    inner_block.add_module("fc2".to_string(), Linear::new(6, 4).unwrap());

    // Create outer sequential model
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input_layer".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("inner_block".to_string(), inner_block);
    model.add_module("output_layer".to_string(), Linear::new(4, 2).unwrap());

    // Test forward pass
    let input = TestTensor::ones(&[1, 10]).unwrap();
    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape().dims(), &[1, 2]);

    // Verify nested structure
    assert_eq!(model.modules().len(), 3); // input_layer, inner_block, output_layer
}

/// Test parameter collection from nested modules
#[test]
fn test_parameter_collection_nested() {
    // Create nested structure
    let mut inner1 = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner1.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    inner1.add_module("relu".to_string(), ReLU::new());

    let mut inner2 = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner2.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());
    inner2.add_module("gelu".to_string(), GeLU::new());

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("block1".to_string(), inner1);
    model.add_module("block2".to_string(), inner2);
    model.add_module("output".to_string(), Linear::new(2, 1).unwrap());

    // Collect all parameters
    let params = model.parameters();

    // Should have 3 Linear layers × 2 params each = 6 params
    assert_eq!(params.len(), 6);

    // Verify all parameters require gradients
    for param in params.iter() {
        assert!(param.requires_grad());
    }
}

/// Test parameter collection consistency
#[test]
fn test_parameter_collection_consistency() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(5, 4).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(4, 3).unwrap());

    // Collect parameters multiple times
    let params1 = model.parameters();
    let params2 = model.parameters();

    // Should return same number of parameters
    assert_eq!(params1.len(), params2.len());
    assert_eq!(params1.len(), 4); // 2 Linear layers × 2 params each

    // Parameter names should be consistent
    for (p1, p2) in params1.iter().zip(params2.iter()) {
        assert_eq!(p1.name(), p2.name());
    }
}

/// Test module traversal
#[test]
fn test_module_traversal() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("relu1".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(8, 6).unwrap());
    model.add_module("relu2".to_string(), ReLU::new());

    // Get all modules
    let modules = model.modules();
    assert_eq!(modules.len(), 4);

    // Verify module names
    assert_eq!(modules[0].name(), "Linear");
    assert_eq!(modules[1].name(), "ReLU");
    assert_eq!(modules[2].name(), "Linear");
    assert_eq!(modules[3].name(), "ReLU");
}

/// Test deep nested structure
#[test]
fn test_deep_nested_structure() {
    // Create deeply nested structure
    let mut level3 = Sequential::<TestBackend, TestStorage, Float32>::new();
    level3.add_module("fc".to_string(), Linear::new(3, 2).unwrap());
    level3.add_module("relu".to_string(), ReLU::new());

    let mut level2 = Sequential::<TestBackend, TestStorage, Float32>::new();
    level2.add_module("fc".to_string(), Linear::new(4, 3).unwrap());
    level2.add_module("block".to_string(), level3);

    let mut level1 = Sequential::<TestBackend, TestStorage, Float32>::new();
    level1.add_module("fc".to_string(), Linear::new(5, 4).unwrap());
    level1.add_module("block".to_string(), level2);

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input".to_string(), Linear::new(6, 5).unwrap());
    model.add_module("deep_block".to_string(), level1);
    model.add_module("output".to_string(), Linear::new(2, 1).unwrap());

    // Test forward pass through deep structure
    let input = TestTensor::ones(&[1, 6]).unwrap();
    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape().dims(), &[1, 1]);

    // Verify parameter collection works through deep nesting
    let params = model.parameters();
    assert_eq!(params.len(), 10); // 5 Linear layers × 2 params each
}

/// Test parameter collection with mixed layer types
#[test]
fn test_parameter_collection_mixed_layers() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();

    // Add layers with different parameter counts
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap()); // 2 params
    model.add_module("relu".to_string(), ReLU::new()); // 0 params
    model.add_module("fc2".to_string(), Linear::new(8, 6).unwrap()); // 2 params
    model.add_module("gelu".to_string(), GeLU::new()); // 0 params
    model.add_module("fc3".to_string(), Linear::new(6, 4).unwrap()); // 2 params

    let params = model.parameters();

    // Should only collect parameters from Linear layers
    assert_eq!(params.len(), 6); // 3 Linear layers × 2 params each

    // Verify all collected parameters are valid
    for param in params.iter() {
        assert!(param.requires_grad());
        assert!(!param.data().as_slice().is_empty());
    }
}

/// Test zero_grad propagation through nested modules
#[test]
fn test_zero_grad_nested_propagation() {
    let mut inner = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    inner.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input".to_string(), Linear::new(5, 4).unwrap());
    model.add_module("inner".to_string(), inner);
    model.add_module("output".to_string(), Linear::new(2, 1).unwrap());

    // Verify parameters exist before zero_grad
    let params_before = model.parameters();
    assert_eq!(params_before.len(), 8); // 4 Linear layers × 2 params each

    // All parameters should initially require gradients
    for param in params_before.iter() {
        assert!(param.requires_grad());
    }

    // Call zero_grad on outer model
    model.zero_grad();

    // Verify parameters still exist and are accessible after zero_grad
    let params_after = model.parameters();
    assert_eq!(params_after.len(), 8); // 4 Linear layers × 2 params each

    // After zero_grad, parameters are detached (no longer require gradients)
    // This is the expected behavior - zero_grad resets gradient computation
    for param in params_after.iter() {
        assert!(!param.requires_grad());
    }
}

/// Test train/eval mode propagation
#[test]
fn test_train_eval_mode_propagation() {
    let mut inner = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner.add_module("fc1".to_string(), Linear::new(3, 2).unwrap());
    inner.add_module("relu".to_string(), ReLU::new());

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("inner".to_string(), inner);

    let input = TestTensor::ones(&[1, 4]).unwrap();

    // Test in train mode
    model.train(true);
    let output_train = model.forward(&input).unwrap();
    assert_eq!(output_train.shape().dims(), &[1, 2]);

    // Test in eval mode
    model.train(false);
    let output_eval = model.forward(&input).unwrap();
    assert_eq!(output_eval.shape().dims(), &[1, 2]);

    // For Linear + ReLU, outputs should be deterministic
    // (no dropout or batch norm that behaves differently)
}

/// Test empty Sequential container
#[test]
fn test_empty_sequential() {
    let model = Sequential::<TestBackend, TestStorage, Float32>::new();

    assert_eq!(model.modules().len(), 0);
    assert_eq!(model.parameters().len(), 0);
    assert_eq!(model.name(), "Sequential");
}

/// Test Sequential with single module
#[test]
fn test_sequential_single_module() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc".to_string(), Linear::new(5, 3).unwrap());

    let input = TestTensor::ones(&[2, 5]).unwrap();
    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape().dims(), &[2, 3]);
    assert_eq!(model.modules().len(), 1);
    assert_eq!(model.parameters().len(), 2); // weight + bias
}

/// Test parameter names are preserved
#[test]
fn test_parameter_names_preserved() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("layer1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("layer2".to_string(), Linear::new(3, 2).unwrap());

    let params = model.parameters();

    // Verify parameter names exist and are non-empty
    for param in params.iter() {
        assert!(!param.name().is_empty());
    }
}

/// Test module composition with batch processing
#[test]
fn test_composition_batch_processing() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(8, 4).unwrap());

    // Test with different batch sizes
    for batch_size in [1, 2, 4, 8, 16] {
        let input = TestTensor::ones(&[batch_size, 10]).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[batch_size, 4]);
    }
}

/// Test gradient flow through nested composition
#[test]
fn test_gradient_flow_nested_composition() {
    let mut inner = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    inner.add_module("relu".to_string(), ReLU::new());

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input".to_string(), Linear::new(5, 4).unwrap());
    model.add_module("inner".to_string(), inner);
    model.add_module("output".to_string(), Linear::new(3, 2).unwrap());

    let input = TestTensor::ones(&[1, 5]).unwrap().requires_grad_(true);
    let output = model.forward(&input).unwrap();

    // Verify gradient tracking is preserved
    assert!(output.requires_grad());

    // Verify all parameters require gradients
    let params = model.parameters();
    for param in params.iter() {
        assert!(param.requires_grad());
    }
}
