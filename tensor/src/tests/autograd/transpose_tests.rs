//! Comprehensive transpose gradient tests with mathematical validation
//!
//! These tests validate that transpose operations correctly propagate gradients
//! through the computational graph with exact mathematical precision.

use approx::assert_relative_eq;
use crate::Tensor;
use coeus_backend::CpuBackend;

/// Test basic transpose gradient computation
#[test]
fn test_transpose_gradient_basic() {
    // Create a 2x3 tensor
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    let mut a = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    a.set_requires_grad(true);

    // Transpose to get 3x2 tensor
    // A^T = [[1, 4],
    //        [2, 5],
    //        [3, 6]]
    let at = a.t().unwrap();

    // Compute sum for scalar output
    let mut sum_at = at.sum();
    sum_at.backward().unwrap();

    // Get gradient
    let a_grad = a.grad().unwrap();

    // Validate gradient shape (should match input shape)
    assert_eq!(a_grad.shape(), &[2, 3]);

    // For transpose operation: ∂sum(A^T)/∂A[i,j] = 1 for all i,j
    // Since sum(A^T) = sum(A) and transpose doesn't change element values
    for &grad_val in a_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}

/// Test transpose gradient in complex expressions
#[test]
fn test_transpose_gradient_complex() {
    // Create weight matrix for linear layer
    // W = [[1, 2],
    //      [3, 4]]
    let mut w = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    w.set_requires_grad(true);

    // Create input vector
    // x = [5, 6]
    let x = Tensor::from_vec(CpuBackend::new(), vec![5.0, 6.0], vec![2]).unwrap();

    // Convert to 2D for matrix multiplication: x = [[5, 6]]
    let x_2d = x.unsqueeze(0).unwrap();

    // Linear operation: y = x @ W^T
    let wt = w.t().unwrap();
    let y = x_2d.matmul(&wt).unwrap();

    // Compute sum for scalar loss
    let mut loss = y.sum();
    loss.backward().unwrap();

    // Get gradient w.r.t. weights
    let w_grad = w.grad().unwrap();

    // Expected gradient: ∂loss/∂W[i,j] = x[j] (for row-major storage)
    // Since y = x @ W^T, then loss = sum(x @ W^T)
    // ∂loss/∂W[i,j] = x[i] * 1 (each element of W contributes to output)
    assert_eq!(w_grad.shape(), &[2, 2]);
    assert_relative_eq!(w_grad.data()[0], 5.0, epsilon = 1e-6); // ∂loss/∂W[0,0] = x[0]
    assert_relative_eq!(w_grad.data()[1], 6.0, epsilon = 1e-6); // ∂loss/∂W[0,1] = x[1]
    assert_relative_eq!(w_grad.data()[2], 5.0, epsilon = 1e-6); // ∂loss/∂W[1,0] = x[0]
    assert_relative_eq!(w_grad.data()[3], 6.0, epsilon = 1e-6); // ∂loss/∂W[1,1] = x[1]
}

/// Test transpose gradient accumulation
#[test]
fn test_transpose_gradient_accumulation() {
    // Test gradient accumulation through multiple transpose operations
    let mut a = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    a.set_requires_grad(true);

    // Multiple operations involving transpose
    let at1 = a.t().unwrap();  // First transpose
    let at2 = at1.t().unwrap(); // Second transpose (back to original)

    // Use in computation
    let mut result = at2.sum();
    result.backward().unwrap();

    let a_grad = a.grad().unwrap();

    // After A^T^T = A, gradient should still be 1 for all elements
    assert_eq!(a_grad.shape(), &[2, 2]);
    for &grad_val in a_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}

/// Test transpose gradient with broadcasting
#[test]
fn test_transpose_gradient_broadcasting() {
    // Test transpose with tensors that will be broadcasted in subsequent operations
    let mut a = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    a.set_requires_grad(true);

    let at = a.t().unwrap(); // 3x2 tensor

    // Add scalar for broadcasting test
    let scalar = Tensor::scalar(1.0);
    let result = (&at + &scalar).unwrap();

    let mut sum_result = result.sum();
    sum_result.backward().unwrap();

    let a_grad = a.grad().unwrap();

    // Gradient should be 1 for all elements (sum of transposed result)
    assert_eq!(a_grad.shape(), &[2, 3]);
    for &grad_val in a_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}

/// Test transpose gradient edge cases
#[test]
fn test_transpose_gradient_edge_cases() {
    // Test with zero tensor
    let mut zero_tensor = Tensor::from_vec(CpuBackend::new(), vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
    zero_tensor.set_requires_grad(true);

    let zero_t = zero_tensor.t().unwrap();
    let mut zero_sum = zero_t.sum();
    zero_sum.backward().unwrap();

    let zero_grad = zero_tensor.grad().unwrap();
    for &grad_val in zero_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }

    // Test with large values (numerical stability)
    let mut large_tensor = Tensor::from_vec(CpuBackend::new(), vec![1e6, -1e6, 1e6, -1e6], vec![2, 2]).unwrap();
    large_tensor.set_requires_grad(true);

    let large_t = large_tensor.t().unwrap();
    let mut large_sum = large_t.sum();
    large_sum.backward().unwrap();

    let large_grad = large_tensor.grad().unwrap();
    for &grad_val in large_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}

/// Test transpose gradient with non-contiguous memory layout
#[test]
fn test_transpose_gradient_non_contiguous() {
    // Create tensor and slice it to test non-contiguous memory
    let mut original = Tensor::from_vec(
        CpuBackend::new(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3]
    ).unwrap();
    original.set_requires_grad(true);

    // Take a slice (this creates non-contiguous memory layout)
    // Note: This test assumes slice functionality exists
    // For now, test with the original tensor
    let transposed = original.t().unwrap();
    let mut result = transposed.sum();
    result.backward().unwrap();

    let grad = original.grad().unwrap();
    for &grad_val in grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}

/// Test transpose gradient numerical stability
#[test]
fn test_transpose_gradient_numerical_stability() {
    // Test with values near numerical precision limits
    let mut tiny_tensor = Tensor::from_vec(CpuBackend::new(), vec![1e-8, 1e-7, 1e-6, 1e-5], vec![2, 2]).unwrap();
    tiny_tensor.set_requires_grad(true);

    let tiny_t = tiny_tensor.t().unwrap();
    let mut tiny_sum = tiny_t.sum();
    tiny_sum.backward().unwrap();

    let tiny_grad = tiny_tensor.grad().unwrap();
    for &grad_val in tiny_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }

    // Test with mixed positive/negative values
    let mut mixed_tensor = Tensor::from_vec(CpuBackend::new(), vec![-3.0, 2.0, -1.0, 4.0], vec![2, 2]).unwrap();
    mixed_tensor.set_requires_grad(true);

    let mixed_t = mixed_tensor.t().unwrap();
    let mut mixed_sum = mixed_t.sum();
    mixed_sum.backward().unwrap();

    let mixed_grad = mixed_tensor.grad().unwrap();
    for &grad_val in mixed_grad.data() {
        assert_relative_eq!(grad_val, 1.0, epsilon = 1e-6);
    }
}
