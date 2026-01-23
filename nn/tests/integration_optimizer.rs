//! Integration Tests for Optimizer Integration
//!
//! Tests optimizer with various layers, state_dict/load_state_dict, and learning rate scheduling.
//! Validates Requirements 15.2

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, Module, ReLU, Sequential};
use optim::{
    Adam, BaseOptimizer, CosineAnnealingLR, ExponentialLR, LRScheduler, Optimizer, RMSprop, StepLR,
    SGD,
};
use storage::DenseStorage;
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

/// Test optimizer with Linear layer
#[test]
fn test_optimizer_with_linear_layer() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(5, 3).unwrap();

    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Test zero_grad
    BaseOptimizer::zero_grad(&mut optimizer);

    // Test learning rate access
    assert_eq!(optimizer.get_lr(), 0.01);

    // Test parameter groups
    assert_eq!(optimizer.param_groups().len(), 1);
}

/// Test optimizer with multiple layer types
#[test]
fn test_optimizer_with_multiple_layers() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(10, 8).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(8, 4).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    // Should have 4 parameters (2 Linear layers × 2 params each)
    assert_eq!(param_tensors.len(), 4);

    let mut optimizer = Adam::new(param_tensors, 0.001);

    // Test optimizer operations
    BaseOptimizer::zero_grad(&mut optimizer);
    assert_eq!(optimizer.get_lr(), 0.001);
}

/// Test optimizer with deep network
#[test]
fn test_optimizer_with_deep_network() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(20, 16).unwrap());
    model.add_module("relu1".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(16, 12).unwrap());
    model.add_module("relu2".to_string(), ReLU::new());
    model.add_module("fc3".to_string(), Linear::new(12, 8).unwrap());
    model.add_module("relu3".to_string(), ReLU::new());
    model.add_module("fc4".to_string(), Linear::new(8, 4).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    // Should have 8 parameters (4 Linear layers × 2 params each)
    assert_eq!(param_tensors.len(), 8);

    let mut optimizer = Adam::new(param_tensors, 0.001);

    // Test multiple zero_grad calls
    for _ in 0..5 {
        BaseOptimizer::zero_grad(&mut optimizer);
    }

    assert_eq!(optimizer.param_groups().len(), 1);
}

/// Test different optimizer types with same model
#[test]
fn test_different_optimizer_types() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(4, 2).unwrap();
    let params = layer.parameters();

    // Test Adam
    {
        let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();
        let mut adam = Adam::new(param_tensors, 0.001);
        assert_eq!(adam.name(), "Adam");
        assert_eq!(adam.get_lr(), 0.001);
        BaseOptimizer::zero_grad(&mut adam);
    }

    // Test SGD
    {
        let sgd = SGD::<TestBackend, Float32>::new(0.01, 0.0, 0.0, 0.0, false);
        assert_eq!(sgd.name(), "SGD");
        assert_eq!(sgd.lr(), 0.01);
    }

    // Test RMSprop
    {
        let mut rmsprop =
            RMSprop::<TestBackend, TestStorage, Float32>::new(0.01, 0.9, 1e-8, 0.0, 0.0, false);
        assert_eq!(rmsprop.name(), "RMSprop");
        BaseOptimizer::zero_grad(&mut rmsprop);
    }
}

/// Test optimizer with parameter groups
#[test]
fn test_optimizer_parameter_groups() {
    let layer1 = Linear::<TestBackend, TestStorage, Float32>::new(5, 4).unwrap();
    let layer2 = Linear::<TestBackend, TestStorage, Float32>::new(4, 2).unwrap();

    let params1 = layer1.parameters();
    let params2 = layer2.parameters();

    let param_tensors1: Vec<TestTensor> = params1.iter().map(|p| p.data().clone()).collect();
    let param_tensors2: Vec<TestTensor> = params2.iter().map(|p| p.data().clone()).collect();

    // Create optimizer with first group
    let mut optimizer = Adam::new(param_tensors1, 0.01);

    // Add second parameter group
    optimizer.add_param_group(param_tensors2);

    // Should have 2 parameter groups
    assert_eq!(optimizer.param_groups().len(), 2);
}

/// Test learning rate scheduling with StepLR
#[test]
fn test_step_lr_scheduling() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.1);

    // Initial learning rate
    assert_eq!(optimizer.get_lr(), 0.1);

    // Create StepLR scheduler (step every 5 epochs, multiply by 0.5)
    let mut scheduler = StepLR::new(0.1, 5, 0.5);

    // Step 4 times (should not change LR yet)
    for _ in 0..4 {
        scheduler.step();
    }
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    assert_eq!(optimizer.get_lr(), 0.1);

    // 5th step should change LR
    scheduler.step();
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    assert_eq!(optimizer.get_lr(), 0.05); // 0.1 * 0.5

    // Another 5 steps
    for _ in 0..5 {
        scheduler.step();
    }
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    assert_eq!(optimizer.get_lr(), 0.025); // 0.05 * 0.5
}

/// Test learning rate scheduling with ExponentialLR
#[test]
fn test_exponential_lr_scheduling() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.1);

    // Create ExponentialLR scheduler (gamma = 0.9)
    let mut scheduler = ExponentialLR::new(0.1, 0.9);

    // Step once
    scheduler.step();
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    let lr1 = optimizer.get_lr();
    assert!((lr1 - 0.09).abs() < 1e-6); // 0.1 * 0.9

    // Step again
    scheduler.step();
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    let lr2 = optimizer.get_lr();
    assert!((lr2 - 0.081).abs() < 1e-6); // 0.09 * 0.9
}

/// Test learning rate scheduling with CosineAnnealingLR
#[test]
fn test_cosine_annealing_lr_scheduling() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.1);

    // Create CosineAnnealingLR scheduler (10 steps, min_lr = 0.0)
    let mut scheduler = CosineAnnealingLR::new(0.1, 0.0, 10);

    // Initial LR
    assert_eq!(optimizer.get_lr(), 0.1);

    // Step through half the cycle
    for _ in 0..5 {
        scheduler.step();
    }
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    let mid_lr = optimizer.get_lr();

    // At midpoint, LR should be around 0.05 (halfway between 0.1 and 0.0)
    assert!(mid_lr > 0.04 && mid_lr < 0.06);

    // Step to the end
    for _ in 0..5 {
        scheduler.step();
    }
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    let final_lr = optimizer.get_lr();

    // At the end, LR should be close to min_lr (0.0)
    assert!(final_lr < 0.01);
}

/// Test optimizer zero_grad consistency
#[test]
fn test_optimizer_zero_grad_consistency() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Call zero_grad multiple times
    for _ in 0..10 {
        BaseOptimizer::zero_grad(&mut optimizer);
    }

    // Optimizer should still be functional
    assert_eq!(optimizer.param_groups().len(), 1);
    assert_eq!(optimizer.get_lr(), 0.01);
}

/// Test optimizer with nested Sequential containers
#[test]
fn test_optimizer_with_nested_containers() {
    let mut inner = Sequential::<TestBackend, TestStorage, Float32>::new();
    inner.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    inner.add_module("relu".to_string(), ReLU::new());

    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("input".to_string(), Linear::new(5, 4).unwrap());
    model.add_module("inner".to_string(), inner);
    model.add_module("output".to_string(), Linear::new(3, 2).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    // Should collect all parameters from nested structure
    assert_eq!(param_tensors.len(), 6); // 3 Linear layers × 2 params each

    let mut optimizer = Adam::new(param_tensors, 0.001);

    BaseOptimizer::zero_grad(&mut optimizer);
    assert_eq!(optimizer.param_groups().len(), 1);
}

/// Test learning rate updates
#[test]
fn test_learning_rate_updates() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Initial LR
    assert_eq!(optimizer.get_lr(), 0.01);

    // Update LR
    BaseOptimizer::set_lr(&mut optimizer, 0.001);
    assert_eq!(optimizer.get_lr(), 0.001);

    // Update again
    BaseOptimizer::set_lr(&mut optimizer, 0.0001);
    assert_eq!(optimizer.get_lr(), 0.0001);
}

/// Test optimizer with batch processing
#[test]
fn test_optimizer_batch_processing() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(10, 5).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(5, 2).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.01);

    // Simulate multiple batches
    for _ in 0..5 {
        // Forward pass (simulated)
        let input = TestTensor::ones(&[4, 10]).unwrap();
        let _output = model.forward(&input).unwrap();

        // Zero gradients
        BaseOptimizer::zero_grad(&mut optimizer);
    }

    // Optimizer should still be functional
    assert_eq!(optimizer.param_groups().len(), 1);
}

/// Test scheduler with multiple epochs
#[test]
fn test_scheduler_multiple_epochs() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.1);
    let mut scheduler = StepLR::new(0.1, 2, 0.5);

    let mut learning_rates = Vec::new();

    // Simulate 10 epochs
    for _ in 0..10 {
        learning_rates.push(optimizer.get_lr());
        scheduler.step();
        BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    }

    // Verify LR changes at expected intervals
    assert_eq!(learning_rates[0], 0.1);
    assert_eq!(learning_rates[1], 0.1);
    // After 2 steps, LR should change
    assert!(learning_rates[2] < 0.1);
}

/// Test optimizer name consistency
#[test]
fn test_optimizer_name_consistency() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let adam = Adam::new(param_tensors, 0.01);
    assert_eq!(adam.name(), "Adam");

    let sgd = SGD::<TestBackend, Float32>::new(0.01, 0.0, 0.0, 0.0, false);
    assert_eq!(sgd.name(), "SGD");

    let rmsprop =
        RMSprop::<TestBackend, TestStorage, Float32>::new(0.01, 0.9, 1e-8, 0.0, 0.0, false);
    assert_eq!(rmsprop.name(), "RMSprop");
}

/// Test optimizer with empty parameter list
#[test]
fn test_optimizer_empty_parameters() {
    let empty_params: Vec<TestTensor> = vec![];
    let optimizer = Adam::new(empty_params, 0.01);

    // Should create optimizer with no parameters
    assert_eq!(optimizer.param_groups().len(), 1);
    assert_eq!(optimizer.get_lr(), 0.01);
}

/// Test learning rate scheduler edge cases
#[test]
fn test_scheduler_edge_cases() {
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params = layer.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.1);

    // Test StepLR with step_size = 1 (changes every step)
    let mut scheduler = StepLR::new(0.1, 1, 0.5);
    scheduler.step();
    BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    assert_eq!(optimizer.get_lr(), 0.05); // 0.1 * 0.5

    // Test ExponentialLR with gamma = 1.0 (no change)
    let layer2 = Linear::<TestBackend, TestStorage, Float32>::new(3, 2).unwrap();
    let params2 = layer2.parameters();
    let param_tensors2: Vec<TestTensor> = params2.iter().map(|p| p.data().clone()).collect();
    let mut optimizer2 = Adam::new(param_tensors2, 0.1);

    let mut scheduler2 = ExponentialLR::new(0.1, 1.0);
    scheduler2.step();
    BaseOptimizer::set_lr(&mut optimizer2, scheduler2.learning_rate() as f32);
    assert_eq!(optimizer2.get_lr(), 0.1); // Should remain the same
}

/// Test optimizer integration with training loop structure
#[test]
fn test_optimizer_training_loop_integration() {
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    let params = model.parameters();
    let param_tensors: Vec<TestTensor> = params.iter().map(|p| p.data().clone()).collect();

    let mut optimizer = Adam::new(param_tensors, 0.01);
    let mut scheduler = StepLR::new(0.01, 5, 0.5);

    // Simulate training loop
    for epoch in 0..10 {
        // Verify LR at start of epoch (before stepping scheduler)
        if epoch < 5 {
            assert_eq!(optimizer.get_lr(), 0.01);
        } else {
            assert_eq!(optimizer.get_lr(), 0.005);
        }

        // Simulate batch processing
        for _ in 0..3 {
            let input = TestTensor::ones(&[2, 4]).unwrap();
            let _output = model.forward(&input).unwrap();

            // Zero gradients
            BaseOptimizer::zero_grad(&mut optimizer);
        }

        // Step scheduler at end of epoch
        scheduler.step();
        BaseOptimizer::set_lr(&mut optimizer, scheduler.learning_rate() as f32);
    }
}
