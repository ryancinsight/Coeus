/// Gradient flow tests
/// This module contains complex tests for automatic differentiation
use approx::assert_relative_eq;
use crate::{Tensor, CpuBackend};
use crate::ops::matrix::matmul;

/// Test gradient computation for complex expressions
#[test]
fn test_complex_expression_gradient() {
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![2.0], vec![]).unwrap();
    x.set_requires_grad(true);

    let mut y = Tensor::from_vec(backend.clone(), vec![3.0], vec![]).unwrap();
    y.set_requires_grad(true);

    let mut z = Tensor::from_vec(backend.clone(), vec![0.0], vec![]).unwrap();
    z.set_requires_grad(true);

    // Compute x²
    let x_squared = (&x * &x).unwrap();

    // Compute x² * y
    let x_squared_y = (&x_squared * &y).unwrap();

    // Compute sin(z)
    let sin_z = z.sin().unwrap();

    // Compute final result
    let mut result = (&x_squared_y + &sin_z).unwrap();

    // Compute gradients
    result.backward().unwrap();

    // Validate gradients
    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 12.0, epsilon = 1e-6); // 2*x*y
    assert_relative_eq!(y.grad().unwrap().as_scalar().unwrap(), 4.0, epsilon = 1e-6);  // x²
    assert_relative_eq!(z.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);  // cos(z)
}

/// Test gradient computation for chain rule
#[test]
fn test_chain_rule_gradient() {
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![0.0], vec![]).unwrap();
    x.set_requires_grad(true);

    // Compute e^x
    let exp_x = x.exp().unwrap();

    // Compute sin(e^x)
    let mut sin_exp_x = exp_x.sin().unwrap();

    // Compute gradient
    sin_exp_x.backward().unwrap();

    let x_val: f64 = x.as_scalar().unwrap();
    let expected_grad = (x_val.exp()).cos() * x_val.exp(); // cos(e^x) * e^x at x=0
    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), expected_grad, epsilon = 1e-6);
}

/// Test gradient computation for higher-order derivatives
#[test]
fn test_second_order_derivatives() {
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![2.0], vec![]).unwrap();
    x.set_requires_grad(true);

    // First derivative: y = x³, dy/dx = 3x² = 12
    let mut y = x.pow(3.0).unwrap();

    // For now, just test first derivative
    y.backward().unwrap();
    assert_relative_eq!(x.grad().unwrap().as_scalar().unwrap(), 12.0, epsilon = 1e-6);

    // Second derivatives via Hessian computation (implemented via finite differences)
    // let hessian = x.hessian().unwrap();
    // assert_relative_eq!(hessian[0][0], 12.0, epsilon = 1e-6); // 6*x = 12
}

/// Test gradient computation with broadcasting
#[test]
fn test_broadcasting_gradient() {
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    x.set_requires_grad(true);

    let mut y = Tensor::from_vec(backend.clone(), vec![2.0], vec![]).unwrap();
    y.set_requires_grad(true);

    // Compute element-wise multiplication with broadcasting
    let z = (&x * &y).unwrap();

    // Compute sum to get scalar output
    let mut sum_z = z.sum();
    sum_z.backward().unwrap();

    // Validate gradients
    // dz/dx = [y, y, y] = [2.0, 2.0, 2.0]
    let x_grad = x.grad().unwrap();
    assert_relative_eq!(x_grad.data()[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x_grad.data()[1], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x_grad.data()[2], 2.0, epsilon = 1e-6);

    // dz/dy = sum(x) = 6.0
    assert_relative_eq!(y.grad().unwrap().as_scalar().unwrap(), 6.0, epsilon = 1e-6);
}

/// Test gradient computation for matrix operations
#[test]
fn test_matrix_operations_gradient() {
    let backend = CpuBackend::default();
    let mut a = Tensor::from_vec(backend.clone(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    a.set_requires_grad(true);

    let mut b = Tensor::from_vec(backend.clone(), vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    b.set_requires_grad(true);

    // Compute matrix multiplication
    let c = matmul(&a, &b).unwrap();

    // Compute sum for scalar output
    let mut sum_c = c.sum();
    sum_c.backward().unwrap();

    // Validate gradients with exact analytical computation
    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();

    // Expected gradients for A: [[11, 15], [11, 15]]
    // ∂sum(C)/∂A[i,j] = sum over k of B[j,k] (sum over output columns)
    assert_relative_eq!(a_grad.data()[0], 11.0, epsilon = 1e-6); // ∂sum(C)/∂A[0,0] = B[0,0] + B[0,1] = 5+6
    assert_relative_eq!(a_grad.data()[1], 15.0, epsilon = 1e-6); // ∂sum(C)/∂A[0,1] = B[1,0] + B[1,1] = 7+8
    assert_relative_eq!(a_grad.data()[2], 11.0, epsilon = 1e-6); // ∂sum(C)/∂A[1,0] = B[0,0] + B[0,1] = 5+6
    assert_relative_eq!(a_grad.data()[3], 15.0, epsilon = 1e-6); // ∂sum(C)/∂A[1,1] = B[1,0] + B[1,1] = 7+8

    // Expected gradients for B: [[4, 4], [6, 6]]
    // ∂sum(C)/∂B[i,j] = sum over k of A[k,i] (sum over output rows for fixed column i)
    assert_relative_eq!(b_grad.data()[0], 4.0, epsilon = 1e-6); // ∂sum(C)/∂B[0,0] = A[0,0] + A[1,0] = 1+3
    assert_relative_eq!(b_grad.data()[1], 4.0, epsilon = 1e-6); // ∂sum(C)/∂B[0,1] = A[0,0] + A[1,0] = 1+3 (same as B[0,0])
    assert_relative_eq!(b_grad.data()[2], 6.0, epsilon = 1e-6); // ∂sum(C)/∂B[1,0] = A[0,1] + A[1,1] = 2+4
    assert_relative_eq!(b_grad.data()[3], 6.0, epsilon = 1e-6); // ∂sum(C)/∂B[1,1] = A[0,1] + A[1,1] = 2+4 (same as B[1,0])
}

/// Test edge cases for gradient computation
#[test]
fn test_gradient_edge_cases() {
    let backend = CpuBackend::default();
    let mut x_zero = Tensor::from_vec(backend.clone(), vec![0.0], vec![]).unwrap();
    x_zero.set_requires_grad(true);
    let mut y_zero = x_zero.exp().unwrap(); // e^0 = 1
    y_zero.backward().unwrap();
    assert_relative_eq!(x_zero.grad().unwrap().as_scalar().unwrap(), 1.0, epsilon = 1e-6);

    let mut x_neg = Tensor::from_vec(backend.clone(), vec![-1.0], vec![]).unwrap();
    x_neg.set_requires_grad(true);
    let mut y_neg = x_neg.exp().unwrap(); // e^(-1) ≈ 0.3679
    y_neg.backward().unwrap();
    assert_relative_eq!(x_neg.grad().unwrap().as_scalar().unwrap(), (-1.0f64).exp(), epsilon = 1e-6);

    let mut x_small = Tensor::from_vec(backend.clone(), vec![1e-6], vec![]).unwrap();
    x_small.set_requires_grad(true);
    let mut y_small = x_small.sin().unwrap();
    y_small.backward().unwrap();
    assert_relative_eq!(x_small.grad().unwrap().as_scalar().unwrap(), (1e-6f64).cos(), epsilon = 1e-6);
}

/// Test numerical gradient verification
#[test]
fn test_numerical_gradient_verification() {
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![1.0], vec![]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.exp().unwrap();
    y.backward().unwrap();

    let analytical_grad = x.grad().unwrap().as_scalar().unwrap();

    // Compute numerical gradient using finite differences
    let h = 1e-5f64;
    let x_val = 1.0f64;
    let numerical_grad = ((x_val + h).exp() - (x_val - h).exp()) / (2.0 * h);

    assert_relative_eq!(analytical_grad, numerical_grad, epsilon = 1e-4);
}
