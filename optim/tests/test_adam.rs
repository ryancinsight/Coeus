//! Tests for Adam optimizer.
//!
//! Validates Adam implementation against PyTorch behavior:
//! - Basic gradient descent with adaptive learning rates
//! - Bias correction for first and second moments
//! - Multiple optimization steps with moment accumulation
//! - Custom hyperparameters (beta1, beta2, epsilon)
//! - Edge cases (zero gradients, numerical stability)

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use coeus_optim::{Optimizer, Adam};

type TestTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

/// Helper to create a tensor with requires_grad=true
fn tensor_with_grad(data: Vec<Float32>, dims: &[usize]) -> TestTensor {
    TestTensor::from_vec(data, dims).unwrap().requires_grad_(true)
}

#[test]
fn test_adam_basic() {
    // Test basic Adam with default hyperparameters
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)], &[3]);
    let grad = TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)], &[3]).unwrap();
    
    // Set gradient manually (simulating backward pass)
    param.set_grad(grad).unwrap();
    
    let mut optimizer = Adam::default(0.001).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 1);
    
    // After first step with bias correction, parameters should be updated
    // The exact values depend on bias correction: m_hat = m / (1 - 0.9^1) = m / 0.1
    // v_hat = v / (1 - 0.999^1) = v / 0.001
    let data = param.as_slice();
    
    // Parameters should have decreased (gradient descent)
    assert!(data[0].get() < 1.0);
    assert!(data[1].get() < 2.0);
    assert!(data[2].get() < 3.0);
}

#[test]
fn test_adam_bias_correction() {
    // Test that bias correction works correctly on first few steps
    let mut param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    
    let mut optimizer = Adam::default(0.01).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    // First step
    param.set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap()).unwrap();
    optimizer.step().unwrap();
    
    let val1 = param.as_slice()[0].get();
    
    // Second step
    param.set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap()).unwrap();
    optimizer.step().unwrap();
    
    let val2 = param.as_slice()[0].get();
    
    // Third step
    param.set_grad(TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap()).unwrap();
    optimizer.step().unwrap();
    
    let val3 = param.as_slice()[0].get();
    
    // Values should be monotonically decreasing
    assert!(val1 < 1.0);
    assert!(val2 < val1);
    assert!(val3 < val2);
}

#[test]
fn test_adam_multiple_steps() {
    // Test Adam over multiple steps with moment accumulation
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    
    let mut optimizer = Adam::default(0.01).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    // Run 10 steps with constant gradient
    for _ in 0..10 {
        param.set_grad(TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(1.0)], &[2]).unwrap()).unwrap();
        optimizer.step().unwrap();
    }
    
    // After 10 steps, parameters should have decreased significantly
    let data = param.as_slice();
    assert!(data[0].get() < 0.95, "Expected param[0] < 0.95, got {}", data[0].get());
    assert!(data[1].get() < 1.95, "Expected param[1] < 1.95, got {}", data[1].get());
}

#[test]
fn test_adam_custom_hyperparams() {
    // Test Adam with custom hyperparameters
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let grad = TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap();
    param.set_grad(grad).unwrap();
    
    // Custom: beta1=0.95, beta2=0.99, epsilon=1e-7
    let mut optimizer = Adam::new(0.001, 0.95, 0.99, 1e-7).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    optimizer.step().unwrap();
    
    // Parameters should be updated
    let data = param.as_slice();
    assert!(data[0].get() < 1.0);
    assert!(data[1].get() < 2.0);
}

#[test]
fn test_adam_zero_grad() {
    // Test that zero_grad clears gradients
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    param.set_grad(TestTensor::from_vec(vec![Float32::new(0.5), Float32::new(0.5)], &[2]).unwrap()).unwrap();
    
    let mut optimizer = Adam::default(0.001).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    optimizer.zero_grad();
    
    // Gradient should be cleared
    assert!(param.grad().is_err());
}

#[test]
fn test_adam_no_grad() {
    // Test that parameters without gradients are skipped
    let mut param = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    // Don't set gradient
    
    let mut optimizer = Adam::default(0.001).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 0); // No parameters updated
    
    // Parameter should be unchanged
    let data = param.as_slice();
    assert_eq!(data[0].get(), 1.0);
    assert_eq!(data[1].get(), 2.0);
}

#[test]
fn test_adam_learning_rate() {
    // Test learning rate getter/setter
    let mut optimizer: Adam<CpuBackend, DenseStorage<Float32>, Float32> = Adam::default(0.001).unwrap();
    assert_eq!(optimizer.learning_rate(), 0.001);
    
    optimizer.set_learning_rate(0.01).unwrap();
    assert_eq!(optimizer.learning_rate(), 0.01);
    
    // Test invalid learning rate
    assert!(optimizer.set_learning_rate(0.0).is_err());
    assert!(optimizer.set_learning_rate(-0.1).is_err());
}

#[test]
fn test_adam_requires_grad() {
    // Test that parameters without requires_grad are rejected
    let mut param = TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    // Don't set requires_grad
    
    let mut optimizer = Adam::default(0.001).unwrap();
    assert!(optimizer.add_param(&mut param).is_err());
}

#[test]
fn test_adam_multiple_params() {
    // Test optimizer with multiple parameters
    let mut param1 = tensor_with_grad(vec![Float32::new(1.0), Float32::new(2.0)], &[2]);
    let mut param2 = tensor_with_grad(vec![Float32::new(3.0), Float32::new(4.0)], &[2]);
    
    param1.set_grad(TestTensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap()).unwrap();
    param2.set_grad(TestTensor::from_vec(vec![Float32::new(0.3), Float32::new(0.4)], &[2]).unwrap()).unwrap();
    
    let mut optimizer = Adam::default(0.01).unwrap();
    optimizer.add_param(&mut param1).unwrap();
    optimizer.add_param(&mut param2).unwrap();
    
    let updated = optimizer.step().unwrap();
    assert_eq!(updated, 2);
    
    // Both parameters should be updated
    let data1 = param1.as_slice();
    assert!(data1[0].get() < 1.0);
    assert!(data1[1].get() < 2.0);
    
    let data2 = param2.as_slice();
    assert!(data2[0].get() < 3.0);
    assert!(data2[1].get() < 4.0);
}

#[test]
fn test_adam_convergence() {
    // Test that Adam converges to minimum for simple quadratic
    // f(x) = x^2, gradient = 2x, minimum at x=0
    let mut param = tensor_with_grad(vec![Float32::new(10.0)], &[1]);

    // Use higher learning rate for faster convergence
    let mut optimizer = Adam::default(0.5).unwrap();
    optimizer.add_param(&mut param).unwrap();

    // Run 200 steps (Adam needs more steps than SGD due to adaptive rates)
    for _ in 0..200 {
        // Compute gradient: 2x
        let x = param.as_slice()[0].get();
        let grad = TestTensor::from_vec(vec![Float32::new(2.0 * x)], &[1]).unwrap();
        param.set_grad(grad).unwrap();

        optimizer.step().unwrap();
    }

    // Should converge close to 0
    let final_x = param.as_slice()[0].get();
    assert!(final_x.abs() < 0.5, "Adam should converge to minimum, got {}", final_x);
}

#[test]
fn test_adam_numerical_stability() {
    // Test that epsilon prevents division by zero
    let mut param = tensor_with_grad(vec![Float32::new(1.0)], &[1]);
    
    // Use very small epsilon to test numerical stability
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-10).unwrap();
    optimizer.add_param(&mut param).unwrap();
    
    // Set very small gradient
    param.set_grad(TestTensor::from_vec(vec![Float32::new(1e-8)], &[1]).unwrap()).unwrap();
    
    // Should not panic or produce NaN
    optimizer.step().unwrap();
    
    let val = param.as_slice()[0].get();
    assert!(val.is_finite(), "Parameter should remain finite");
    assert!(val < 1.0, "Parameter should decrease");
}

#[test]
fn test_adam_invalid_hyperparams() {
    // Test that invalid hyperparameters are rejected
    
    // Invalid learning rate
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.0, 0.9, 0.999, 1e-8).is_err());
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(-0.1, 0.9, 0.999, 1e-8).is_err());
    
    // Invalid beta1
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 0.0, 0.999, 1e-8).is_err());
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 1.0, 0.999, 1e-8).is_err());
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 1.1, 0.999, 1e-8).is_err());
    
    // Invalid beta2
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 0.9, 0.0, 1e-8).is_err());
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 0.9, 1.0, 1e-8).is_err());
    assert!(Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 0.9, 1.1, 1e-8).is_err());
}

