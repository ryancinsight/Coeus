/// Activation function gradient tests
/// This module contains tests for activation functions and their gradients
use approx::assert_relative_eq;
use crate::Tensor;

/// Test gradient computation for power function
#[test]
fn test_power_gradient() {
    // Test: f(x) = x^n, f'(x) = n * x^(n-1)
    // At x=2.0, n=3.0: f'(x) = 3 * 2^(3-1) = 3 * 4 = 12.0
    let mut x = Tensor::scalar(2.0);
    x.set_requires_grad(true);

    let y = x.pow(3.0);
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 12.0, epsilon = 1e-6);
}

/// Test gradient computation for exponential function
#[test]
fn test_exp_gradient() {
    // Test: f(x) = e^x, f'(x) = e^x
    // At x=0.0: f'(x) = e^0 = 1.0
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    let y = x.exp();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);
}

/// Test gradient computation for logarithm
#[test]
fn test_log_gradient() {
    // Test: f(x) = ln(x), f'(x) = 1/x
    // At x=1.0: f'(x) = 1/1 = 1.0
    let mut x = Tensor::scalar(1.0);
    x.set_requires_grad(true);

    let y = x.log();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);
}

/// Test gradient computation for sine function
#[test]
fn test_sin_gradient() {
    // Test: f(x) = sin(x), f'(x) = cos(x)
    // At x=0.0: f'(x) = cos(0) = 1.0
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    let y = x.sin();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);
}

/// Test gradient computation for cosine function
#[test]
fn test_cos_gradient() {
    // Test: f(x) = cos(x), f'(x) = -sin(x)
    // At x=0.0: f'(x) = -sin(0) = 0.0
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    let y = x.cos();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 0.0, epsilon = 1e-6);
}

/// Test gradient computation for ReLU
#[test]
fn test_relu_gradient() {
    // Test ReLU function: f(x) = max(0, x), f'(x) = 1 if x > 0, 0 if x <= 0

    // Test positive input
    let mut x_pos = Tensor::scalar(1.0);
    x_pos.set_requires_grad(true);

    let y_pos = x_pos.relu();
    y_pos.backward().unwrap();

    assert_relative_eq!(x_pos.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);

    // Test negative input
    let mut x_neg = Tensor::scalar(-1.0);
    x_neg.set_requires_grad(true);

    let y_neg = x_neg.relu();
    y_neg.backward().unwrap();

    assert_relative_eq!(x_neg.grad().unwrap().as_scalar().unwrap(), 0.0, epsilon = 1e-6);

    // Test edge case at exactly zero (critical point where derivative is mathematically undefined)
    // PyTorch convention: ReLU'(0) = 0
    let mut x_zero = Tensor::scalar(0.0);
    x_zero.set_requires_grad(true);

    let y_zero = x_zero.relu();
    y_zero.backward().unwrap();

    assert_relative_eq!(x_zero.grad().unwrap().as_scalar().unwrap(), 0.0, epsilon = 1e-6);
}

/// Test gradient computation for sigmoid function
#[test]
fn test_sigmoid_gradient() {
    // Test: f(x) = 1/(1+e^(-x)), f'(x) = f(x) * (1 - f(x))
    // At x=0.0: f(x) = 0.5, f'(x) = 0.5 * (1-0.5) = 0.25
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    let y = x.sigmoid();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 0.25, epsilon = 1e-6);
}

/// Test gradient computation for tanh function
#[test]
fn test_tanh_gradient() {
    // Test: f(x) = tanh(x), f'(x) = 1 - tanh²(x)
    // At x=0.0: f(x) = 0.0, f'(x) = 1 - 0 = 1.0
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    let y = x.tanh();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);
}

/// Test gradient computation for square root function
#[test]
fn test_sqrt_gradient() {
    // Test: f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
    // At x=4.0: f'(x) = 1/(2*sqrt(4)) = 1/(2*2) = 0.25
    let mut x = Tensor::scalar(4.0);
    x.set_requires_grad(true);

    let y = x.sqrt();
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 0.25, epsilon = 1e-6);
}

/// Test ceil function
#[test]
fn test_ceil() {
    let tensor = Tensor::from_vec(vec![1.3, -1.7, 2.9, 2.0], vec![4]);
    let ceil_tensor = tensor.ceil();

    assert_eq!(ceil_tensor.data()[0], 2.0);
    assert_eq!(ceil_tensor.data()[1], -1.0);
    assert_eq!(ceil_tensor.data()[2], 3.0);
    assert_eq!(ceil_tensor.data()[3], 2.0);
}

/// Test floor function
#[test]
fn test_floor() {
    let tensor = Tensor::from_vec(vec![1.3, -1.7, 2.9, 2.0], vec![4]);
    let floor_tensor = tensor.floor();

    assert_eq!(floor_tensor.data()[0], 1.0);
    assert_eq!(floor_tensor.data()[1], -2.0);
    assert_eq!(floor_tensor.data()[2], 2.0);
    assert_eq!(floor_tensor.data()[3], 2.0);
}

/// Test round function
#[test]
fn test_round() {
    let tensor = Tensor::from_vec(vec![1.3, 1.7, 2.5, 2.0, 1.5], vec![5]);
    let round_tensor = tensor.round();

    assert_eq!(round_tensor.data()[0], 1.0);
    assert_eq!(round_tensor.data()[1], 2.0);
    assert_eq!(round_tensor.data()[2], 3.0); // Round half up
    assert_eq!(round_tensor.data()[3], 2.0);
    assert_eq!(round_tensor.data()[4], 2.0); // Round half up
}

/// Test square function
#[test]
fn test_square() {
    let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0, -2.0], vec![4]);
    let square_tensor = tensor.square();

    assert_eq!(square_tensor.data()[0], 4.0);
    assert_eq!(square_tensor.data()[1], 9.0);
    assert_eq!(square_tensor.data()[2], 16.0);
    assert_eq!(square_tensor.data()[3], 4.0); // (-2)^2 = 4
}

/// Test reciprocal function
#[test]
fn test_reciprocal() {
    let tensor = Tensor::from_vec(vec![2.0, 4.0, 0.5, -2.0], vec![4]);
    let reciprocal_tensor = tensor.reciprocal();

    assert_eq!(reciprocal_tensor.data()[0], 0.5);
    assert_eq!(reciprocal_tensor.data()[1], 0.25);
    assert_eq!(reciprocal_tensor.data()[2], 2.0);
    assert_eq!(reciprocal_tensor.data()[3], -0.5); // 1/(-2) = -0.5
}

/// Test sign function
#[test]
fn test_sign() {
    let tensor = Tensor::from_vec(vec![-2.0, 0.0, 3.0, -0.1, 0.1], vec![5]);
    let sign_tensor = tensor.sign();

    assert_eq!(sign_tensor.data()[0], -1.0);
    assert_eq!(sign_tensor.data()[1], 0.0);
    assert_eq!(sign_tensor.data()[2], 1.0);
    assert_eq!(sign_tensor.data()[3], -1.0);
    assert_eq!(sign_tensor.data()[4], 1.0);
}

/// Test that mathematical operations preserve requires_grad flag
#[test]
fn test_math_ops_requires_grad() {
    let mut tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
    tensor.set_requires_grad(true);

    let ceil_tensor = tensor.ceil();
    let floor_tensor = tensor.floor();
    let round_tensor = tensor.round();
    let square_tensor = tensor.square();
    let reciprocal_tensor = tensor.reciprocal();
    let sign_tensor = tensor.sign();

    assert!(ceil_tensor.requires_grad());
    assert!(floor_tensor.requires_grad());
    assert!(round_tensor.requires_grad());
    assert!(square_tensor.requires_grad());
    assert!(reciprocal_tensor.requires_grad());
    assert!(sign_tensor.requires_grad());
}