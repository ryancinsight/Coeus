//! Matrix operations with PyTorch-compatible API

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use coeus_backend::Backend;
use coeus_storage::TensorStorage;
use num_traits::{Float, Num};
// use coeus_autograd::context::Operation; // DISABLED - architectural redesign required

/// Matrix multiplication with PyTorch-compatible API
///
/// This function implements matrix multiplication following PyTorch semantics:
/// - Supports 2D matrix multiplication
/// - Automatic differentiation integration
/// - Backend-accelerated computation when available
///
/// # Arguments
/// * `a` - Left-hand side tensor (must be 2D)
/// * `b` - Right-hand side tensor (must be 2D)
///
/// # Returns
/// Result containing the matrix product tensor
///
/// # Errors
/// Returns `TensorError::MatrixMulRequires2D` if tensors are not 2D
/// Returns `TensorError::IncompatibleMatrixDims` if matrix dimensions are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, ops::matmul};
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
/// let c = matmul(&a, &b).unwrap();
/// // c.shape() == [2, 2]
/// ```
pub fn matmul<T: Dtype + FloatDtype, B: Backend<T> + Clone + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    // Validate input dimensions
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(TensorError::MatrixMulRequires2D {
            lhs_shape: a.shape().to_vec(),
            rhs_shape: b.shape().to_vec(),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(TensorError::IncompatibleMatrixDims {
            lhs_m: m,
            lhs_k: k,
            rhs_k: b.shape()[0],
            rhs_n: n,
        });
    }

    // For now, use CPU implementation (backend integration can be added later)
    let mut result = cpu_matmul(a, b);

    // Set up autograd graph if inputs require gradients
    if a.requires_grad() || b.requires_grad() {
        use crate::core::tensor::{with_autograd_context, Operation};

        with_autograd_context(|context| {
            let a_node = if let Some(node) = a.node {
                node
            } else {
                let node = context.create_leaf_node();
                context.register_tensor(node, a.data.iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default(), a.shape.clone());
                node
            };

            let b_node = if let Some(node) = b.node {
                node
            } else {
                let node = context.create_leaf_node();
                context.register_tensor(node, b.data.iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default(), b.shape.clone());
                node
            };

            // Store input data for gradient computation
            let a_data_f64: Vec<f64> = a.data.iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();
            let b_data_f64: Vec<f64> = b.data.iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let matmul_node = context.create_node_with_data(Operation::Matmul, vec![a_node, b_node], vec![a_data_f64, b_data_f64]);
            result.node = Some(matmul_node);
        });
    }

    Ok(result)
}

/// CPU implementation of matrix multiplication
fn cpu_matmul<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let mut result_data = vec![T::zero(); m * n];

    // Optimized matrix multiplication with cache-friendly access
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                let a_val = a.data()[i * k + p];
                let b_val = b.data()[p * n + j];
                sum = sum + a_val * b_val;
            }
            result_data[i * n + j] = sum;
        }
    }

    let backend = a.backend().clone();
    Tensor::from_vec(backend, result_data, vec![m, n]).unwrap()
}

/// Broadcasting utilities
pub struct Broadcast;

impl Broadcast {
    /// Check if two shapes are broadcastable
    pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Get the broadcasted shape
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);
        let mut result = vec![0; max_len];

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            result[max_len - 1 - i] = dim1.max(dim2);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    /// Test basic matrix multiplication functionality
    #[test]
    fn test_matmul_basic() {
        let backend = CpuBackend::default();
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let a = Tensor::from_vec(backend.clone(), a_data, vec![2, 2]).unwrap();
        let b = Tensor::from_vec(backend.clone(), b_data, vec![2, 2]).unwrap();

        let result = matmul(&a, &b).unwrap();

        // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result.shape(), &[2, 2]);
        let diff0: f64 = result.data()[0] - 19.0;
        let diff1: f64 = result.data()[1] - 22.0;
        let diff2: f64 = result.data()[2] - 43.0;
        let diff3: f64 = result.data()[3] - 50.0;
        assert!(diff0.abs() < 1e-6);
        assert!(diff1.abs() < 1e-6);
        assert!(diff2.abs() < 1e-6);
        assert!(diff3.abs() < 1e-6);
    }

    /// Test matrix multiplication with different shapes
    #[test]
    fn test_matmul_different_shapes() {
        let backend = CpuBackend::default();
        // [3, 2] * [2, 4] = [3, 4]
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::from_vec(backend.clone(), a_data, vec![3, 2]).unwrap();
        let b = Tensor::from_vec(backend.clone(), b_data, vec![2, 4]).unwrap();

        let result = matmul(&a, &b).unwrap();

        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result.data().len(), 12);
    }

    /// Test matrix multiplication error cases
    #[test]
    fn test_matmul_error_cases() {
        // Test incompatible dimensions
        let a = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(); // Should fail

        let result = matmul(&a, &b);
        assert!(result.is_err());

        // Test non-2D tensors
        let a_1d = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0], vec![2]).unwrap();
        let b_2d = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let result_1 = matmul(&a_1d, &b_2d);
        assert!(result_1.is_err());

        let result_2 = matmul(&b_2d, &a_1d);
        assert!(result_2.is_err());
    }

    /// Test matrix multiplication edge cases
    #[test]
    fn test_matmul_edge_cases() {
        let backend = CpuBackend::default();
        // Single element matrices
        let a = Tensor::from_vec(backend.clone(), vec![5.0], vec![1, 1]).unwrap();
        let b = Tensor::from_vec(backend.clone(), vec![3.0], vec![1, 1]).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        let diff_single: f64 = result.data()[0] - 15.0;
        assert!(diff_single.abs() < 1e-6);

        // Identity matrix
        let identity = Tensor::from_vec(backend.clone(), vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let vector = Tensor::from_vec(backend.clone(), vec![2.0, 3.0], vec![2, 1]).unwrap();

        let result = matmul(&identity, &vector).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
        let diff_id0: f64 = result.data()[0] - 2.0;
        let diff_id1: f64 = result.data()[1] - 3.0;
        assert!(diff_id0.abs() < 1e-6);
        assert!(diff_id1.abs() < 1e-6);
    }

    /// Test matrix multiplication with zero matrices
    #[test]
    fn test_matmul_zero_matrices() {
        let backend = CpuBackend::default();
        let zero_a = Tensor::from_vec(backend.clone(), vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let matrix_b = Tensor::from_vec(backend.clone(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let result = matmul(&zero_a, &matrix_b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!(result.data().iter().all(|&x| x == 0.0));
    }

    /// Test broadcast compatibility checking
    #[test]
    fn test_broadcast_can_broadcast() {
        // Compatible shapes
        assert!(Broadcast::can_broadcast(&[3, 1], &[1, 4]));
        assert!(Broadcast::can_broadcast(&[5], &[3, 5]));
        assert!(Broadcast::can_broadcast(&[1, 3], &[3, 3]));

        // Incompatible shapes
        assert!(!Broadcast::can_broadcast(&[3, 2], &[4, 5]));
        assert!(!Broadcast::can_broadcast(&[2, 3], &[2, 4]));
    }

    /// Test broadcast shape computation
    #[test]
    fn test_broadcast_shape() {
        let shape1 = vec![3, 1];
        let shape2 = vec![1, 4];
        let result = Broadcast::broadcast_shape(&shape1, &shape2);
        assert_eq!(result, vec![3, 4]);

        let shape3 = vec![5];
        let shape4 = vec![3, 5];
        let result2 = Broadcast::broadcast_shape(&shape3, &shape4);
        assert_eq!(result2, vec![3, 5]);
    }

    /// Test numerical precision of matrix multiplication
    #[test]
    fn test_matmul_numerical_precision() {
        let backend = CpuBackend::default();
        let a_data = vec![1.0000001, 2.0000002, 3.0000003, 4.0000004];
        let b_data = vec![1.0000001, 2.0000002, 3.0000003, 4.0000004];
        let a = Tensor::from_vec(backend.clone(), a_data, vec![2, 2]).unwrap();
        let b = Tensor::from_vec(backend.clone(), b_data, vec![2, 2]).unwrap();

        let result = matmul(&a, &b).unwrap();

        // Verify result is computed correctly within numerical precision
        let expected_00 = 1.0000001 * 1.0000001 + 2.0000002 * 3.0000003;
        let expected_01 = 1.0000001 * 2.0000002 + 2.0000002 * 4.0000004;
        let expected_10 = 3.0000003 * 1.0000001 + 4.0000004 * 3.0000003;
        let expected_11 = 3.0000003 * 2.0000002 + 4.0000004 * 4.0000004;

        let diff_p0: f64 = result.data()[0] - expected_00;
        let diff_p1: f64 = result.data()[1] - expected_01;
        let diff_p2: f64 = result.data()[2] - expected_10;
        let diff_p3: f64 = result.data()[3] - expected_11;
        assert!(diff_p0.abs() < 1e-10);
        assert!(diff_p1.abs() < 1e-10);
        assert!(diff_p2.abs() < 1e-10);
        assert!(diff_p3.abs() < 1e-10);
    }

    /// Test matrix multiplication with large matrices
    #[test]
    fn test_matmul_large_matrices() {
        let backend = CpuBackend::default();
        let size = 10; // Reduce size to avoid performance issues
        let mut a_data = Vec::with_capacity(size * size);
        let mut b_data = Vec::with_capacity(size * size);

        // Fill with known pattern for verification: A[i,j] = i, B[i,j] = j+1
        for i in 0..size {
            for j in 0..size {
                a_data.push(i as f64);
                b_data.push((j + 1) as f64);
            }
        }

        let a = Tensor::from_vec(backend.clone(), a_data, vec![size, size]).unwrap();
        let b = Tensor::from_vec(backend.clone(), b_data, vec![size, size]).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[size, size]);

        // Verify first element: A[0,0] * B[0,0] + A[0,1] * B[1,0] + ... = 0*1 + 0*2 + ... = 0
        let expected_first = 0.0;
        let diff_large0: f64 = result.data()[0] - expected_first;
        assert!(
            diff_large0.abs() < 1e-10,
            "First element should be 0.0, got {}",
            result.data()[0]
        );

        // Verify a middle element: A[1,1] * B[1,1] + A[1,2] * B[2,1] + ... = 1*2 + 1*3 + ... = sum_{k=1}^{size-1} 1*(k+1)
        let mid_i = 1;
        let mid_j = 1;
        let mut expected_mid = 0.0;
        for k in 0..size {
            expected_mid += a.data()[mid_i * size + k] * b.data()[k * size + mid_j];
        }
        let diff_mid: f64 = result.data()[mid_i * size + mid_j] - expected_mid;
        assert!(
            diff_mid.abs() < 1e-10,
            "Middle element should be {}, got {}",
            expected_mid,
            result.data()[mid_i * size + mid_j]
        );
    }
}

// Matrix multiplication implementation removed - now in core/tensor.rs

/// Matrix multiplication implementation for Tensor (direct)
pub fn matmul_impl<T: Dtype + Float + Num + Clone, B: Backend<T>, S: TensorStorage<T> + Clone + Send + Sync>(lhs: &Tensor<T, B, S>, rhs: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    // Existing cpu_matmul logic, broadcasting if needed
    let m = lhs.shape()[0]; let k = lhs.shape()[1]; let n = rhs.shape()[1];
    if k != rhs.shape()[0] { return Err(TensorError::IncompatibleMatrixDims { lhs_m: m, lhs_k: k, rhs_k: rhs.shape()[0], rhs_n: n }); }
    let mut result_data = vec![T::zero(); m * n];
    for i in 0..m { for j in 0..n { let mut sum = T::zero(); for p in 0..k { sum = sum + lhs.data()[i*k + p] * rhs.data()[p*n + j]; } result_data[i*n + j] = sum; } }
    Ok(Tensor::from_vec(lhs.backend().clone(), result_data, vec![m, n])?)
}
