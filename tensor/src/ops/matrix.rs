//! Matrix operations

use crate::{FloatDtype, Result, Tensor, TensorError};

/// Matrix multiplication
pub fn matmul<T: FloatDtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(TensorError::InvalidOperation {
            message: "Matrix multiplication requires 2D tensors".to_string(),
        });
    }

    if a.shape()[1] != b.shape()[0] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.shape()[0], b.shape()[1]],
            actual: vec![a.shape()[0], a.shape()[1]],
        });
    }

    let m = a.shape()[0];
    let n = b.shape()[1];
    let k = a.shape()[1];

    let mut result_data = vec![T::zero(); m * n];

    // Matrix multiplication implementation
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                let a_idx = i * k + p;
                let b_idx = p * n + j;
                sum = sum + a.data()[a_idx] * b.data()[b_idx];
            }
            result_data[i * n + j] = sum;
        }
    }

    let mut result = Tensor::from_vec(result_data, vec![m, n]);

    // Handle gradient computation
    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
    }

    Ok(result)
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
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let a = Tensor::from_vec(a_data, vec![2, 2]);
        let b = Tensor::from_vec(b_data, vec![2, 2]);

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
        // [3, 2] * [2, 4] = [3, 4]
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::from_vec(a_data, vec![3, 2]);
        let b = Tensor::from_vec(b_data, vec![2, 4]);

        let result = matmul(&a, &b).unwrap();

        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result.data().len(), 12);
    }

    /// Test matrix multiplication error cases
    #[test]
    fn test_matmul_error_cases() {
        // Test incompatible dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]); // Should fail

        let result = matmul(&a, &b);
        assert!(result.is_err());

        // Test non-2D tensors
        let a_1d = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let result_1 = matmul(&a_1d, &b_2d);
        assert!(result_1.is_err());

        let result_2 = matmul(&b_2d, &a_1d);
        assert!(result_2.is_err());
    }

    /// Test matrix multiplication edge cases
    #[test]
    fn test_matmul_edge_cases() {
        // Single element matrices
        let a = Tensor::from_vec(vec![5.0], vec![1, 1]);
        let b = Tensor::from_vec(vec![3.0], vec![1, 1]);

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        let diff_single: f64 = result.data()[0] - 15.0;
        assert!(diff_single.abs() < 1e-6);

        // Identity matrix
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let vector = Tensor::from_vec(vec![2.0, 3.0], vec![2, 1]);

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
        let zero_a = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let matrix_b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

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
        let a_data = vec![1.0000001, 2.0000002, 3.0000003, 4.0000004];
        let b_data = vec![1.0000001, 2.0000002, 3.0000003, 4.0000004];
        let a = Tensor::from_vec(a_data, vec![2, 2]);
        let b = Tensor::from_vec(b_data, vec![2, 2]);

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

        let a = Tensor::from_vec(a_data, vec![size, size]);
        let b = Tensor::from_vec(b_data, vec![size, size]);

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
