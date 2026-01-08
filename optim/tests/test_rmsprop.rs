//! Tests for RMSprop optimizer.
//!
//! Validates RMSprop implementation including GPU acceleration framework:
//! - Basic RMSprop gradient descent with adaptive learning rates
//! - Centered RMSprop variant
//! - Weight decay functionality
//! - GPU acceleration framework (CPU fallback)
//! - Multiple optimization steps with running averages
//! - Custom hyperparameters (alpha, epsilon)

use backend::CpuBackend;
use dtype::float::Float32;
use optim::gpu_backend::GpuAcceleratedOptimizer;
use optim::{Optimizer, RMSprop};
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Helper to create a tensor with requires_grad=true
fn tensor_with_grad(data: Vec<Float32>, dims: &[usize]) -> TestTensor {
    TestTensor::from_vec(data, dims)
        .unwrap()
        .requires_grad_(true)
}

#[test]
fn test_rmsprop_basic() {
    // Test basic RMSprop with default hyperparameters
    let param = tensor_with_grad(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    );
    let grad = TestTensor::from_vec(
        vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)],
        &[3],
    )
    .unwrap();

    param.set_grad(grad).unwrap();

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 1);

    // Parameters should have decreased (gradient descent)
    let params = optimizer.parameters();
    let data = params[0].as_slice();

    assert!(data[0].get() < 1.0);
    assert!(data[1].get() < 2.0);
    assert!(data[2].get() < 3.0);
}

#[test]
fn test_rmsprop_multiple_steps() {
    // Test RMSprop over multiple steps with running averages
    let param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    // Run 10 steps with constant gradient
    for _ in 0..10 {
        param
            .set_grad(
                TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap(),
            )
            .unwrap();
        optimizer.step().unwrap();
    }

    // After 10 steps, parameters should have decreased significantly
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!(
        data[0].get() < 0.95,
        "Expected param[0] < 0.95, got {}",
        data[0].get()
    );
    assert!(
        data[1].get() < 1.95,
        "Expected param[1] < 1.95, got {}",
        data[1].get()
    );
}

#[test]
fn test_rmsprop_custom_alpha() {
    // Test RMSprop with custom alpha (smoothing constant)
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
        .unwrap();

    // Alpha = 0.9 (less smoothing than default 0.99)
    let mut optimizer = RMSprop::new(0.01, 0.9, 1e-8, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    optimizer.step().unwrap();

    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!(data[0].get() < 1.0);
}

#[test]
fn test_rmsprop_centered() {
    // Test centered RMSprop variant
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
        .unwrap();

    let mut optimizer_centered = RMSprop::centered_rmsprop(0.01);
    assert!(optimizer_centered.centered());

    optimizer_centered
        .add_param(&mut param.clone(), "centered".to_string())
        .unwrap();
    optimizer_centered.step().unwrap();

    // Parameters should be updated
    let params = optimizer_centered.parameters();
    let data = params[0].as_slice();
    assert!(data[0].get() < 1.0);
}

#[test]
fn test_rmsprop_weight_decay() {
    // Test that weight decay works correctly
    let param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = RMSprop::new(0.01, 0.99, 1e-8, 0.01, 0.0, false); // weight_decay = 0.01
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    let original_data = param.as_slice().to_vec();

    // Set zero gradient to isolate weight decay effect
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.0), Float32::new(0.0)], &[2]).unwrap())
        .unwrap();
    optimizer.step().unwrap();

    // Parameters should decrease due to weight decay
    let params = optimizer.parameters();
    let new_data = params[0].as_slice();
    assert!(new_data[0].get() < original_data[0].get());
    assert!(new_data[1].get() < original_data[1].get());
}

#[test]
fn test_rmsprop_alpha_momentum() {
    // Test alpha and momentum getters
    let optimizer: RMSprop<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        RMSprop::with_momentum(0.01, 0.9);

    assert_eq!(optimizer.alpha(), 0.99); // Default alpha
    assert_eq!(optimizer.momentum(), 0.9); // Custom momentum
    assert!(!optimizer.centered());
}

#[test]
fn test_rmsprop_zero_grad() {
    // Test that zero_grad clears gradients
    let param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.5), Float32::new(0.5)], &[2]).unwrap())
        .unwrap();

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    optimizer.zero_grad();

    // Gradient should be cleared (no grad available)
    assert!(param.grad().is_err());
}

#[test]
fn test_rmsprop_learning_rate() {
    // Test learning rate getter/setter
    let mut optimizer: RMSprop<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        RMSprop::default(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);

    optimizer.set_learning_rate(0.01).unwrap();
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_rmsprop_convergence() {
    // Test that RMSprop converges to minimum for simple quadratic
    // f(x) = x^2, gradient = 2x, minimum at x=0
    let param = tensor_with_grad(vec![Float32::new(5.0)], &[1]);

    let mut optimizer = RMSprop::default(0.1); // Higher learning rate
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    // Run 100 steps
    for _ in 0..100 {
        let params = optimizer.parameters();
        let x = params[0].as_slice()[0].get();
        let grad = TestTensor::from_vec(vec![Float32::new(2.0 * x)], &[1]).unwrap();
        param.set_grad(grad).unwrap();

        optimizer.step().unwrap();
    }

    // Should converge close to 0
    let params = optimizer.parameters();
    let final_x = params[0].as_slice()[0].get();
    assert!(
        final_x.abs() < 1.0,
        "RMSprop should converge to minimum, got {}",
        final_x
    );
}

#[test]
fn test_rmsprop_gpu_acceleration_framework() {
    // Test GPU acceleration framework setup and CPU fallback
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
        .unwrap();

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    // Initially GPU should not be available
    assert!(!optimizer.gpu_available());
    assert!(optimizer.gpu_backend().is_none());
    assert!(optimizer.gpu_config().is_none());

    // Enable GPU (should set flag even though backend is not initialized)
    use optim::gpu_backend::GpuOptimizerConfig;
    let config = GpuOptimizerConfig::default();
    optimizer.set_gpu_config(config);

    // Now GPU should be marked as enabled
    assert!(optimizer.gpu_available());

    // However, the backend is still None in this test environment
    // (in real implementation, it would be initialized)
}

#[test]
fn test_rmsprop_gpu_step_fallback() {
    // Test that GPU step method falls back to CPU
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let original_val = param.as_slice()[0].get();

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    // Set gradient
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
        .unwrap();

    // GPU step should fall back to CPU (since no GPU backend is initialized)
    let updated = optimizer.step_gpu().unwrap();
    assert_eq!(updated, 1);

    // Parameter should have been updated via CPU fallback
    let params = optimizer.parameters();
    let new_val = params[0].as_slice()[0].get();
    assert!(new_val < original_val);
}

#[test]
fn test_rmsprop_step_cpu_method() {
    // Test the explicit CPU step method
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let original_val = param.as_slice()[0].get();

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
        .unwrap();

    // Use explicit CPU step
    let updated = optimizer.step_cpu().unwrap();
    assert_eq!(updated, 1);

    // Parameter should have been updated
    let params = optimizer.parameters();
    let new_val = params[0].as_slice()[0].get();
    assert!(new_val < original_val);
}

#[test]
fn test_rmsprop_state_persistence() {
    // Test state_dict and load_state_dict functionality
    let param1 = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let param2 = tensor_with_grad(vec![Float32::new(2.0)], &[1]);

    let mut optimizer = RMSprop::default(0.01);
    optimizer
        .add_param(&mut param1.clone(), "p1".to_string())
        .unwrap();
    optimizer
        .add_param(&mut param2.clone(), "p2".to_string())
        .unwrap();

    // Take a few steps to build up state
    for _ in 0..3 {
        param1
            .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
            .unwrap();
        param2
            .set_grad(TestTensor::from_vec(vec![Float32::new(0.2)], &[1]).unwrap())
            .unwrap();
        optimizer.step().unwrap();
    }

    let state_dict = optimizer.state_dict();
    let params = optimizer.parameters();
    let param1_after = params[0].as_slice()[0].get();
    let param2_after = params[1].as_slice()[0].get();

    // Create new optimizer and load state
    let param1_new = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let param2_new = tensor_with_grad(vec![Float32::new(2.0)], &[1]);

    let mut optimizer_new = RMSprop::default(0.01);
    optimizer_new
        .add_param(&mut param1_new.clone(), "p1".to_string())
        .unwrap();
    optimizer_new
        .add_param(&mut param2_new.clone(), "p2".to_string())
        .unwrap();

    optimizer_new.load_state_dict(state_dict).unwrap();

    // Parameters should be restored to the same values
    let params_new = optimizer_new.parameters();
    assert_eq!(params_new[0].as_slice()[0].get(), param1_after);
    assert_eq!(params_new[1].as_slice()[0].get(), param2_after);
}

#[test]
fn test_rmsprop_numerical_stability() {
    // Test that epsilon prevents division by zero
    let param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);

    let mut optimizer = RMSprop::new(0.01, 0.99, 1e-10, 0.0, 0.0, false); // Very small epsilon
    optimizer
        .add_param(&mut param.clone(), "test".to_string())
        .unwrap();

    // Set very small gradient
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1e-8)], &[1]).unwrap())
        .unwrap();

    // Should not panic or produce NaN
    optimizer.step().unwrap();

    let params = optimizer.parameters();
    let val = params[0].as_slice()[0].get();
    assert!(val.is_finite(), "Parameter should remain finite");
    assert!(val < 1.0, "Parameter should decrease");
}
