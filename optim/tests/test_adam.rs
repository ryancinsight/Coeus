//! Tests for Adam optimizer.
//!
//! Validates Adam implementation against PyTorch behavior:
//! - Basic gradient descent with adaptive learning rates
//! - Bias correction for first and second moments
//! - Multiple optimization steps with moment accumulation
//! - Custom hyperparameters (beta1, beta2, epsilon)
//! - Edge cases (zero gradients, numerical stability)

use backend::CpuBackend;
use dtype::float::Float32;
use optim::{Adam, BaseOptimizer, Optimizer};
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
fn test_adam_basic() {
    // Test basic Adam with default hyperparameters
    let param = tensor_with_grad(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    );
    let grad = TestTensor::from_vec(
        vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)],
        &[3],
    )
    .unwrap();

    // Set gradient manually (simulating backward pass)
    param.set_grad(grad).unwrap();

    let mut optimizer = Adam::default(0.001);
    optimizer.add_param_group(vec![param.clone()]);

    let updated = BaseOptimizer::step(&mut optimizer).unwrap();
    assert_eq!(updated, 1);

    // After first step with bias correction, parameters should be updated
    // The exact values depend on bias correction: m_hat = m / (1 - 0.9^1) = m / 0.1
    // v_hat = v / (1 - 0.999^1) = v / 0.001
    let params = optimizer.parameters();
    let data = params[0].as_slice();

    // Parameters should have decreased (gradient descent)
    assert!(data[0].get() < 1.0);
    assert!(data[1].get() < 2.0);
    assert!(data[2].get() < 3.0);
}

#[test]
fn test_adam_bias_correction() {
    // Test that bias correction works correctly on first few steps
    let mut param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param.clone()]);

    // First step
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap())
        .unwrap();
    BaseOptimizer::step(&mut optimizer).unwrap();

    let params = optimizer.parameters();
    let val1 = params[0].as_slice()[0].get();

    // Second step
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap())
        .unwrap();
    BaseOptimizer::step(&mut optimizer).unwrap();

    let params = optimizer.parameters();
    let val2 = params[0].as_slice()[0].get();

    // Third step
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap())
        .unwrap();
    BaseOptimizer::step(&mut optimizer).unwrap();

    let params = optimizer.parameters();
    let val3 = params[0].as_slice()[0].get();

    // Values should be monotonically decreasing
    assert!(val1 < 1.0);
    assert!(val2 < val1);
    assert!(val3 < val2);
}

#[test]
fn test_adam_multiple_steps() {
    // Test Adam over multiple steps with moment accumulation
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param.clone()]);

    // Run 10 steps with constant gradient
    for _ in 0..10 {
        param
            .set_grad(
                TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap(),
            )
            .unwrap();
        BaseOptimizer::step(&mut optimizer).unwrap();
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
fn test_adam_custom_hyperparams() {
    // Test Adam with custom hyperparameters
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let grad = TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap();
    param.set_grad(grad).unwrap();

    // Custom: beta1=0.95, beta2=0.99, epsilon=1e-7
    let mut optimizer = Adam::with_hyperparams(vec![], 0.001, 0.95, 0.99, 1e-7, 0.0);
    optimizer.add_param_group(vec![param.clone()]);

    BaseOptimizer::step(&mut optimizer).unwrap();

    // Parameters should be updated
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!(data[0].get() < 1.0);
    assert!(data[1].get() < 2.0);
}

#[test]
fn test_adam_zero_grad() {
    // Test that zero_grad clears gradients
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.5), Float32::new(0.5)], &[2]).unwrap())
        .unwrap();

    let mut optimizer = Adam::default(0.001);
    optimizer.add_param_group(vec![param.clone()]);

    BaseOptimizer::zero_grad(&mut optimizer);

    // Gradient should be cleared
    assert!(param.grad().is_err());
}

#[test]
fn test_adam_no_grad() {
    // Test that parameters without gradients are skipped
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    // Don't set gradient

    let mut optimizer = Adam::default(0.001);
    optimizer.add_param_group(vec![param.clone()]);

    let updated = BaseOptimizer::step(&mut optimizer).unwrap();
    assert_eq!(updated, 0); // No parameters updated

    // Parameter should be unchanged
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert_eq!(data[0].get(), 1.0);
    assert_eq!(data[1].get(), 2.0);
}

#[test]
fn test_adam_learning_rate() {
    // Test learning rate getter/setter
    let mut optimizer: Adam<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Adam::default(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);

    let _ = optimizer.set_learning_rate(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_adam_multiple_params() {
    // Test optimizer with multiple parameters
    let mut param1 = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let mut param2 = tensor_with_grad(vec![Float32::new(3.0), Float32::new(4.0)], &[2]);

    param1
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap())
        .unwrap();
    param2
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.3), Float32::new(0.4)], &[2]).unwrap())
        .unwrap();

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param1.clone(), param2.clone()]);

    let updated = BaseOptimizer::step(&mut optimizer).unwrap();
    assert_eq!(updated, 2);

    // Both parameters should be updated
    let params = optimizer.parameters();
    let data1 = params[0].as_slice();
    assert!(data1[0].get() < 1.0);
    assert!(data1[1].get() < 2.0);

    let data2 = params[1].as_slice();
    assert!(data2[0].get() < 3.0);
    assert!(data2[1].get() < 4.0);
}

#[test]
fn test_adam_convergence() {
    // Test that Adam converges to minimum for simple quadratic
    // f(x) = x^2, gradient = 2x, minimum at x=0
    let mut param = tensor_with_grad(vec![Float32::new(10.0)], &[1]);

    // Use higher learning rate for faster convergence
    let mut optimizer = Adam::default(0.5);
    optimizer.add_param_group(vec![param.clone()]);

    // Run 200 steps (Adam needs more steps than SGD due to adaptive rates)
    for _ in 0..200 {
        // Compute gradient: 2x
        let params = optimizer.parameters();
        let x = params[0].as_slice()[0].get();
        let grad = TestTensor::from_vec(vec![Float32::new(2.0 * x)], &[1]).unwrap();
        param.set_grad(grad).unwrap();

        BaseOptimizer::step(&mut optimizer).unwrap();
    }

    // Should converge close to 0
    let params = optimizer.parameters();
    let final_x = params[0].as_slice()[0].get();
    assert!(
        final_x.abs() < 0.5,
        "Adam should converge to minimum, got {}",
        final_x
    );
}

#[test]
fn test_adam_numerical_stability() {
    // Test that epsilon prevents division by zero
    let mut param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);

    // Use very small epsilon to test numerical stability
    let mut optimizer = Adam::with_hyperparams(vec![], 0.001, 0.9, 0.999, 1e-10, 0.0);
    optimizer.add_param_group(vec![param.clone()]);

    // Set very small gradient
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1e-8)], &[1]).unwrap())
        .unwrap();

    // Should not panic or produce NaN
    BaseOptimizer::step(&mut optimizer).unwrap();

    let params = optimizer.parameters();
    let val = params[0].as_slice()[0].get();
    assert!(val.is_finite(), "Parameter should remain finite");
    assert!(val < 1.0, "Parameter should decrease");
}

#[test]
fn test_adam_zero_gradients() {
    // Test that zero gradients don't change parameters
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let original_data = param.as_slice().to_vec();

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param.clone()]);

    // Set zero gradient
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.0), Float32::new(0.0)], &[2]).unwrap())
        .unwrap();
    BaseOptimizer::step(&mut optimizer).unwrap();

    // Parameters should remain unchanged
    let params = optimizer.parameters();
    let new_data = params[0].as_slice();
    assert_eq!(new_data[0].get(), original_data[0].get());
    assert_eq!(new_data[1].get(), original_data[1].get());
}

#[test]
fn test_adam_large_gradients() {
    // Test numerical stability with large gradients
    let mut param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param.clone()]);

    // Set very large gradient
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1e6)], &[1]).unwrap())
        .unwrap();

    // Should not panic or produce NaN/Infinity
    BaseOptimizer::step(&mut optimizer).unwrap();

    let params = optimizer.parameters();
    let val = params[0].as_slice()[0].get();
    assert!(
        val.is_finite(),
        "Parameter should remain finite with large gradients"
    );
    assert!(
        val < 1.0,
        "Parameter should decrease with large positive gradient"
    );
}

#[test]
fn test_adam_weight_decay() {
    // Test that weight decay works correctly
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = Adam::with_hyperparams(vec![], 0.01, 0.9, 0.999, 1e-8, 0.01); // weight_decay = 0.01
    optimizer.add_param_group(vec![param.clone()]);

    let original_data = param.as_slice().to_vec();

    // Set zero gradient to isolate weight decay effect
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.0), Float32::new(0.0)], &[2]).unwrap())
        .unwrap();
    BaseOptimizer::step(&mut optimizer).unwrap();

    // Parameters should decrease due to weight decay
    let params = optimizer.parameters();
    let new_data = params[0].as_slice();
    assert!(new_data[0].get() < original_data[0].get());
    assert!(new_data[1].get() < original_data[1].get());
}

#[test]
fn test_adam_state_persistence() {
    // Test state_dict and load_state_dict functionality
    let mut param1 = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let mut param2 = tensor_with_grad(vec![Float32::new(2.0)], &[1]);

    let mut optimizer = Adam::default(0.01);
    optimizer.add_param_group(vec![param1.clone(), param2.clone()]);

    // Take a few steps to build up state
    for _ in 0..3 {
        param1
            .set_grad(TestTensor::from_vec(vec![Float32::new(0.1)], &[1]).unwrap())
            .unwrap();
        param2
            .set_grad(TestTensor::from_vec(vec![Float32::new(0.2)], &[1]).unwrap())
            .unwrap();
        BaseOptimizer::step(&mut optimizer).unwrap();
    }

    let state_dict = BaseOptimizer::state_dict(&optimizer);
    let params = optimizer.parameters();
    let param1_after = params[0].as_slice()[0].get();
    let param2_after = params[1].as_slice()[0].get();

    // Create new optimizer and load state
    let mut param1_new = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    let mut param2_new = tensor_with_grad(vec![Float32::new(2.0)], &[1]);

    let mut optimizer_new = Adam::default(0.01);
    optimizer_new.add_param_group(vec![param1_new.clone(), param2_new.clone()]);

    BaseOptimizer::load_state_dict(&mut optimizer_new, state_dict).unwrap();

    // Parameters should be restored to the same values
    let params_new = optimizer_new.parameters();
    assert_eq!(params_new[0].as_slice()[0].get(), param1_after);
    assert_eq!(params_new[1].as_slice()[0].get(), param2_after);
}
