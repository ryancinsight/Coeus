//! Tests for SGD optimizer.
//!
//! Validates SGD implementation against PyTorch behavior:
//! - Basic gradient descent
//! - Momentum accumulation
//! - Weight decay (L2 regularization)
//! - Nesterov acceleration
//! - Edge cases (zero gradients, NaN handling)

use backend::CpuBackend;
use dtype::float::Float32;
use optim::{Optimizer, SGD};
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
fn test_sgd_basic() {
    // Test basic SGD without momentum
    let mut param = tensor_with_grad(
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

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 1);

    // Expected: param = param - lr * grad
    // [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.99).abs() < 1e-6);
    assert!((data[1].get() - 1.98).abs() < 1e-6);
    assert!((data[2].get() - 2.97).abs() < 1e-6);
}

#[test]
fn test_sgd_momentum() {
    // Test SGD with momentum
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = SGD::with_momentum(0.1, 0.9);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    // First step: grad = [1.0, 1.0]
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap())
        .unwrap();
    optimizer.step().unwrap();

    // v = 0.9 * 0 + 1.0 * [1.0, 1.0] = [1.0, 1.0]
    // param = [1.0, 2.0] - 0.1 * [1.0, 1.0] = [0.9, 1.9]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.9).abs() < 1e-6);
    assert!((data[1].get() - 1.9).abs() < 1e-6);

    // Second step: grad = [1.0, 1.0] (same gradient)
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap())
        .unwrap();
    optimizer.step().unwrap();

    // v = 0.9 * [1.0, 1.0] + 1.0 * [1.0, 1.0] = [1.9, 1.9]
    // param = [0.9, 1.9] - 0.1 * [1.9, 1.9] = [0.71, 1.71]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.71).abs() < 1e-6);
    assert!((data[1].get() - 1.71).abs() < 1e-6);
}

#[test]
fn test_sgd_weight_decay() {
    // Test SGD with weight decay (L2 regularization)
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let grad = TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap();
    param.set_grad(grad).unwrap();

    let mut optimizer = SGD::new(0.1, 0.0, 0.01, 0.0, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    optimizer.step().unwrap();

    // grad_with_wd = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    // param = [1.0, 2.0] - 0.1 * [0.11, 0.22] = [0.989, 1.978]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.989).abs() < 1e-6);
    assert!((data[1].get() - 1.978).abs() < 1e-6);
}

#[test]
fn test_sgd_nesterov() {
    // Test SGD with Nesterov momentum
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    let mut optimizer = SGD::new(0.1, 0.9, 0.0, 0.0, true);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    // First step
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap())
        .unwrap();
    optimizer.step().unwrap();

    // v = 0.9 * 0 + 1.0 * [1.0, 1.0] = [1.0, 1.0]
    // update = [1.0, 1.0] + 0.9 * [1.0, 1.0] = [1.9, 1.9] (Nesterov)
    // param = [1.0, 2.0] - 0.1 * [1.9, 1.9] = [0.81, 1.81]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.81).abs() < 1e-6);
    assert!((data[1].get() - 1.81).abs() < 1e-6);
}

#[test]
fn test_sgd_zero_grad() {
    // Test that zero_grad clears gradients
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.5), Float32::new(0.5)], &[2]).unwrap())
        .unwrap();

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    optimizer.zero_grad();

    // Gradient should be cleared
    assert!(param.grad().is_err());
}

#[test]
fn test_sgd_no_grad() {
    // Test that parameters without gradients are skipped
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    // Don't set gradient

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 0); // No parameters updated

    // Parameter should be unchanged
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert_eq!(data[0].get(), 1.0);
    assert_eq!(data[1].get(), 2.0);
}

#[test]
fn test_sgd_learning_rate() {
    // Test learning rate getter/setter
    let mut optimizer: SGD<CpuBackend<Float32>, Float32> = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    assert_eq!(optimizer.learning_rate(), 0.1);

    optimizer.set_learning_rate(0.01).unwrap();
    assert_eq!(optimizer.learning_rate(), 0.01);

    // Test invalid learning rate
    assert!(optimizer.set_learning_rate(0.0).is_err());
    assert!(optimizer.set_learning_rate(-0.1).is_err());
}

#[test]
fn test_sgd_requires_grad() {
    // Test that parameters without requires_grad are rejected
    let mut param = TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    // Don't set requires_grad

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    assert!(optimizer
        .add_param(&mut param, "param".to_string())
        .is_err());
}

#[test]
fn test_sgd_multiple_params() {
    // Test optimizer with multiple parameters
    let mut param1 = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let mut param2 = tensor_with_grad(vec![Float32::new(3.0), Float32::new(4.0)], &[2]);

    param1
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap())
        .unwrap();
    param2
        .set_grad(TestTensor::from_vec(vec![Float32::new(0.3), Float32::new(0.4)], &[2]).unwrap())
        .unwrap();

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param1, "param1".to_string())
        .unwrap();
    optimizer
        .add_param(&mut param2, "param2".to_string())
        .unwrap();

    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 2);

    // Check param1: [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    let params = optimizer.parameters();
    let data1 = params[0].as_slice();
    assert!((data1[0].get() - 0.99).abs() < 1e-6);
    assert!((data1[1].get() - 1.98).abs() < 1e-6);

    // Check param2: [3.0, 4.0] - 0.1 * [0.3, 0.4] = [2.97, 3.96]
    let data2 = params[1].as_slice();
    assert!((data2[0].get() - 2.97).abs() < 1e-6);
    assert!((data2[1].get() - 3.96).abs() < 1e-6);
}

#[test]
fn test_sgd_dampening() {
    // Test SGD with dampening
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);

    // momentum=0.9, dampening=0.5
    let mut optimizer = SGD::new(0.1, 0.9, 0.0, 0.5, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    // First step
    param
        .set_grad(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap())
        .unwrap();
    optimizer.step().unwrap();

    // v = 0.9 * 0 + (1 - 0.5) * [1.0, 1.0] = [0.5, 0.5]
    // param = [1.0, 2.0] - 0.1 * [0.5, 0.5] = [0.95, 1.95]
    let params = optimizer.parameters();
    let data = params[0].as_slice();
    assert!((data[0].get() - 0.95).abs() < 1e-6);
    assert!((data[1].get() - 1.95).abs() < 1e-6);
}

#[test]
fn test_sgd_convergence() {
    // Test that SGD converges to minimum for simple quadratic
    // f(x) = x^2, gradient = 2x, minimum at x=0
    let mut param = tensor_with_grad(vec![Float32::new(10.0)], &[1]);

    let mut optimizer = SGD::new(0.1, 0.0, 0.0, 0.0, false);
    optimizer
        .add_param(&mut param, "param".to_string())
        .unwrap();

    // Run 100 steps
    for _ in 0..100 {
        // Compute gradient: 2x
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
        final_x.abs() < 0.1,
        "SGD should converge to minimum, got {}",
        final_x
    );
}
