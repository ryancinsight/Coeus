/// Activation function gradient tests
/// This module contains tests for activation functions and their gradients
use approx::assert_relative_eq;

/// Test gradient computation for power function
#[test]
fn test_power_gradient() {
    // Test: f(x) = x^n, f'(x) = n * x^(n-1)
    // At x=2.0, n=3.0: f'(x) = 3 * 2^(3-1) = 3 * 4 = 12.0
    let mut x = Tensor::scalar(2.0);
    x.set_requires_grad(true);

    let y = x.pow(3.0);
    y.backward().unwrap();

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 12.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 0.0, epsilon = 1e-6);
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

    assert_relative_eq!(x_pos.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);

    // Test negative input
    let mut x_neg = Tensor::scalar(-1.0);
    x_neg.set_requires_grad(true);

    let y_neg = x_neg.relu();
    y_neg.backward().unwrap();

    assert_relative_eq!(x_neg.grad().unwrap().as_scalar(), 0.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 0.25, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);
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

    assert_relative_eq!(x.grad().unwrap().as_scalar(), 0.25, epsilon = 1e-6);
}
