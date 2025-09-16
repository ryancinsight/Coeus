/// Advanced gradient computation tests
/// This module contains complex tests for automatic differentiation
use approx::assert_relative_eq;
use crate::Tensor;

/// Test gradient computation for complex expressions
#[test]
fn test_complex_expression_gradient() {
    // Test: f(x,y,z) = x² * y + sin(z), ∇f = [2*x*y, x², cos(z)]
    // At x=2.0, y=3.0, z=0.0: ∇f = [12.0, 4.0, 1.0]
    let mut x = Tensor::scalar(2.0);
    x.set_requires_grad(true);

    let mut y = Tensor::scalar(3.0);
    y.set_requires_grad(true);

    let mut z = Tensor::scalar(0.0);
    z.set_requires_grad(true);

    // Compute x²
    let x_squared = (&x * &x).unwrap();

    // Compute x² * y
    let x_squared_y = (&x_squared * &y).unwrap();

    // Compute sin(z)
    let sin_z = z.sin();

    // Compute final result
    let result = (&x_squared_y + &sin_z).unwrap();

    // Compute gradients
    result.backward().unwrap();

    // Validate gradients
    assert_relative_eq!(x.grad().unwrap().as_scalar(), 12.0, epsilon = 1e-6); // 2*x*y
    assert_relative_eq!(y.grad().unwrap().as_scalar(), 4.0, epsilon = 1e-6);  // x²
    assert_relative_eq!(z.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);  // cos(z)
}

/// Test gradient computation for chain rule
#[test]
fn test_chain_rule_gradient() {
    // Test chain rule: f(x) = sin(e^x), f'(x) = cos(e^x) * e^x
    // At x=0.0: f'(x) = cos(1.0) * 1.0 ≈ 0.5403 * 1.0 ≈ 0.5403
    let mut x = Tensor::scalar(0.0);
    x.set_requires_grad(true);

    // Debug: Check initial state
    println!("x requires_grad: {}", x.requires_grad());
    println!("x has node: {}", x.node.is_some());

    // Compute e^x
    let exp_x = x.exp();
    println!("exp_x requires_grad: {}", exp_x.requires_grad());
    println!("exp_x has node: {}", exp_x.node.is_some());

    // Compute sin(e^x)
    let sin_exp_x = exp_x.sin();
    println!("sin_exp_x requires_grad: {}", sin_exp_x.requires_grad());
    println!("sin_exp_x has node: {}", sin_exp_x.node.is_some());

    // Compute gradient
    sin_exp_x.backward().unwrap();

    // Debug: Check final gradients
    println!("x has grad: {}", x.grad().is_some());
    if let Some(grad) = x.grad() {
        println!("x grad value: {}", grad.as_scalar());
    }
    println!("sin_exp_x has grad: {}", sin_exp_x.grad().is_some());
    if let Some(grad) = sin_exp_x.grad() {
        println!("sin_exp_x grad value: {}", grad.as_scalar());
    }

    let expected_grad = (1.0f64).cos() * (0.0f64).exp(); // cos(e^0) * e^0 = cos(1) * 1
    println!("Expected grad: {}", expected_grad);

    assert_relative_eq!(x.grad().unwrap().as_scalar(), expected_grad, epsilon = 1e-6);
}

/// Test gradient computation for higher-order derivatives
#[test]
fn test_second_order_derivatives() {
    // Test second derivative: f(x) = x³, f'(x) = 3x², f''(x) = 6x
    // At x=2.0: f''(x) = 12.0

    // Note: Current implementation may not support second derivatives
    // This test documents the expected behavior for future implementation

    let mut x = Tensor::scalar(2.0);
    x.set_requires_grad(true);

    // First derivative: y = x³, dy/dx = 3x² = 12
    let y = x.pow(3.0);

    // For now, just test first derivative
    y.backward().unwrap();
    assert_relative_eq!(x.grad().unwrap().as_scalar(), 12.0, epsilon = 1e-6);

    // TODO: Implement and test second derivatives when supported
    // let hessian = x.hessian().unwrap();
    // assert_relative_eq!(hessian[0][0], 12.0, epsilon = 1e-6); // 6*x = 12
}

/// Test gradient computation with broadcasting
#[test]
fn test_broadcasting_gradient() {
    // Test gradient computation with broadcasting
    let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    x.set_requires_grad(true);

    let mut y = Tensor::scalar(2.0);
    y.set_requires_grad(true);

    // Compute element-wise multiplication with broadcasting
    let z = (&x * &y).unwrap();

    // Compute sum to get scalar output
    let sum_z = z.sum();
    sum_z.backward().unwrap();

    // Validate gradients
    // dz/dx = [y, y, y] = [2.0, 2.0, 2.0]
    let x_grad = x.grad().unwrap();
    assert_relative_eq!(x_grad.data()[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x_grad.data()[1], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x_grad.data()[2], 2.0, epsilon = 1e-6);

    // dz/dy = sum(x) = 6.0
    assert_relative_eq!(y.grad().unwrap().as_scalar(), 6.0, epsilon = 1e-6);
}

/// Test edge cases for gradient computation
#[test]
fn test_gradient_edge_cases() {
    // Test gradient computation at special values

    // Test at zero
    let mut x_zero = Tensor::scalar(0.0);
    x_zero.set_requires_grad(true);
    let y_zero = x_zero.exp(); // e^0 = 1
    y_zero.backward().unwrap();
    assert_relative_eq!(x_zero.grad().unwrap().as_scalar(), 1.0, epsilon = 1e-6);

    // Test at negative values
    let mut x_neg = Tensor::scalar(-1.0);
    x_neg.set_requires_grad(true);
    let y_neg = x_neg.exp(); // e^(-1) ≈ 0.3679
    y_neg.backward().unwrap();
    assert_relative_eq!(x_neg.grad().unwrap().as_scalar(), (-1.0f64).exp(), epsilon = 1e-6);

    // Test with very small values
    let mut x_small = Tensor::scalar(1e-6);
    x_small.set_requires_grad(true);
    let y_small = x_small.sin();
    y_small.backward().unwrap();
    assert_relative_eq!(x_small.grad().unwrap().as_scalar(), (1e-6f64).cos(), epsilon = 1e-6);
}

/// Test numerical gradient verification
#[test]
fn test_numerical_gradient_verification() {
    // Test that analytical gradients match numerical gradients
    let mut x = Tensor::scalar(1.0);
    x.set_requires_grad(true);

    let y = x.exp();
    y.backward().unwrap();

    let analytical_grad = x.grad().unwrap().as_scalar();

    // Compute numerical gradient using finite differences
    let h = 1e-5_f64;
    let x_val = 1.0_f64;
    let numerical_grad = ((x_val + h).exp() - (x_val - h).exp()) / (2.0_f64 * h);

    assert_relative_eq!(analytical_grad, numerical_grad, epsilon = 1e-4);
}

/// Test gradient computation for matrix operations
#[test]
fn test_matrix_operations_gradient() {
    // Test gradient computation for matrix multiplication with exact validation
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    // sum(C) = 19 + 22 + 43 + 50 = 134
    //
    // ∂sum(C)/∂A[i,j] = sum over k of B[j,k] (since C[i,j] = sum_k A[i,k] * B[k,j])
    // ∂sum(C)/∂B[i,j] = sum over k of A[k,i] (since C[k,j] = sum_m A[k,m] * B[m,j])
    //
    // For A: ∂sum(C)/∂A = [[5+7, 6+8], [5+7, 6+8]] = [[12, 14], [12, 14]]
    // For B: ∂sum(C)/∂B = [[1+3, 2+4], [1+3, 2+4]] = [[4, 6], [4, 6]]

    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    a.set_requires_grad(true);

    let mut b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    b.set_requires_grad(true);

    // Compute matrix multiplication
    let c = a.matmul(&b).unwrap();

    // Compute sum for scalar output
    let sum_c = c.sum();
    sum_c.backward().unwrap();

    // Validate gradients with exact analytical computation
    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();

    // Expected gradients for A: [[11, 15], [11, 15]]
    // ∂sum(C)/∂A[i,j] = sum_l B[j,l] (sum over output columns)
    assert_relative_eq!(a_grad.data()[0], 11.0, epsilon = 1e-6); // ∂sum(C)/∂A[0,0] = B[0,0] + B[0,1] = 5+6
    assert_relative_eq!(a_grad.data()[1], 15.0, epsilon = 1e-6); // ∂sum(C)/∂A[0,1] = B[1,0] + B[1,1] = 7+8
    assert_relative_eq!(a_grad.data()[2], 11.0, epsilon = 1e-6); // ∂sum(C)/∂A[1,0] = B[0,0] + B[0,1] = 5+6
    assert_relative_eq!(a_grad.data()[3], 15.0, epsilon = 1e-6); // ∂sum(C)/∂A[1,1] = B[1,0] + B[1,1] = 7+8

    // Expected gradients for B: [[4, 4], [6, 6]]
    // ∂sum(C)/∂B[i,j] = sum_k A[k,i] (sum over output rows for fixed column i)
    assert_relative_eq!(b_grad.data()[0], 4.0, epsilon = 1e-6); // ∂sum(C)/∂B[0,0] = A[0,0] + A[1,0] = 1+3
    assert_relative_eq!(b_grad.data()[1], 4.0, epsilon = 1e-6); // ∂sum(C)/∂B[0,1] = A[0,0] + A[1,0] = 1+3 (same as B[0,0])
    assert_relative_eq!(b_grad.data()[2], 6.0, epsilon = 1e-6); // ∂sum(C)/∂B[1,0] = A[0,1] + A[1,1] = 2+4
    assert_relative_eq!(b_grad.data()[3], 6.0, epsilon = 1e-6); // ∂sum(C)/∂B[1,1] = A[0,1] + A[1,1] = 2+4 (same as B[1,0])
}
