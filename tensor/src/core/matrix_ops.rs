//! Matrix operations for tensors
//!
//! This module contains matrix multiplication and other linear algebra operations
//! with automatic differentiation support.

use crate::{Tensor, TensorError, Dtype, FloatDtype, Result};
use crate::with_autograd_context;
use coeus_autograd::context::Operation;

#[cfg(test)]
mod matrix_ops_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test basic matrix multiplication
    #[test]
    fn test_matmul_basic() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b).unwrap();
        // [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test matrix multiplication with different shapes
    #[test]
    fn test_matmul_different_shapes() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0], vec![3, 2]);
        let result = a.matmul(&b).unwrap();
        // [[1*7 + 2*9 + 3*8, 1*8 + 2*10 + 3*9], [4*7 + 5*9 + 6*8, 4*8 + 5*10 + 6*9]] = [[50, 56], [113, 128]]
        assert_eq!(result.data(), &[50.0, 56.0, 113.0, 128.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test matrix multiplication edge cases
    #[test]
    fn test_matmul_edge_cases() {
        // Test 1x1 matrices
        let a1x1 = Tensor::from_vec(vec![5.0], vec![1, 1]);
        let b1x1 = Tensor::from_vec(vec![3.0], vec![1, 1]);
        let result1x1 = a1x1.matmul(&b1x1).unwrap();
        assert_eq!(result1x1.data(), &[15.0]);
        assert_eq!(result1x1.shape(), &[1, 1]);

        // Test identity matrix
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let vector = Tensor::from_vec(vec![2.0, 3.0], vec![2, 1]);
        let result_identity = identity.matmul(&vector).unwrap();
        assert_eq!(result_identity.data(), &[2.0, 3.0]);
        assert_eq!(result_identity.shape(), &[2, 1]);

        // Test zero matrix
        let zero_matrix = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let non_zero = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let zero_result = zero_matrix.matmul(&non_zero).unwrap();
        assert_eq!(zero_result.data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(zero_result.shape(), &[2, 2]);
    }

    /// Test matrix multiplication error cases
    #[test]
    fn test_matmul_error_cases() {
        // Test incompatible dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let result = a.matmul(&b);
        assert!(result.is_err());

        // Test 1D tensors (should error)
        let c = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let d = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let result_1d = c.matmul(&d);
        assert!(result_1d.is_err());
    }

    /// Test matrix multiplication with scalar values
    #[test]
    fn test_matmul_scalar() {
        let a = Tensor::from_vec(vec![2.0], vec![1, 1]);
        let b = Tensor::from_vec(vec![3.0], vec![1, 1]);
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.data(), &[6.0]);
        assert_eq!(result.shape(), &[1, 1]);
    }

    /// Test matrix multiplication with identity matrix
    #[test]
    fn test_matmul_identity() {
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = matrix.matmul(&identity).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Test matrix multiplication with zero matrix
    #[test]
    fn test_matmul_zero() {
        let zero = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = matrix.matmul(&zero).unwrap();
        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    /// Test matrix multiplication with gradient computation
    #[test]
    fn test_matmul_with_gradients() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let result = a.matmul(&b).unwrap();
        assert!(result.requires_grad());
        assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    /// Test matrix multiplication numerical precision
    #[test]
    fn test_matmul_numerical_precision() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let result = a.matmul(&b).unwrap();

        // Should be exact match for simple case
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test with floating point precision
        let a_float = Tensor::from_vec(vec![1.1, 2.2, 3.3, 4.4], vec![2, 2]);
        let b_float = Tensor::from_vec(vec![1.0, 0.5, 0.0, 1.0], vec![2, 2]);
        let result_float = a_float.matmul(&b_float).unwrap();

        // Verify computation is correct within tolerance
        assert_relative_eq!(result_float.data()[0], 1.1, epsilon = 1e-10);
        assert_relative_eq!(result_float.data()[1], 2.75, epsilon = 1e-10);
        assert_relative_eq!(result_float.data()[2], 3.3, epsilon = 1e-10);
        assert_relative_eq!(result_float.data()[3], 6.05, epsilon = 1e-10);
    }

    /// Test matrix multiplication edge cases
    #[test]
    fn test_matmul_edge_cases() {
        // Test with large matrices
        let large_a = Tensor::from_vec(vec![1.0; 100], vec![10, 10]);
        let large_b = Tensor::from_vec(vec![1.0; 100], vec![10, 10]);
        let large_result = large_a.matmul(&large_b).unwrap();
        assert_eq!(large_result.shape(), &[10, 10]);

        // Test with minimum size (2x2)
        let min_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let min_b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let min_result = min_a.matmul(&min_b).unwrap();
        assert_eq!(min_result.shape(), &[2, 2]);
    }

    /// Test matrix multiplication associativity
    #[test]
    fn test_matmul_associativity() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]);

        let ab = a.matmul(&b).unwrap();
        let bc = b.matmul(&c).unwrap();
        let abc = ab.matmul(&c).unwrap();

        let ac = a.matmul(&c).unwrap();
        let abc_alt = ac.matmul(&b).unwrap();

        // Matrix multiplication is associative: (AB)C = A(BC)
        assert_eq!(abc.data(), abc_alt.data());
    }

    /// Test matrix multiplication distributivity
    #[test]
    fn test_matmul_distributivity() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let c = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        // Test left distributivity: A(B + C) = AB + AC
        let b_plus_c = b.add(&c).unwrap();
        let a_b_plus_c = a.matmul(&b_plus_c).unwrap();

        let a_b = a.matmul(&b).unwrap();
        let a_c = a.matmul(&c).unwrap();
        let a_b_plus_a_c = a_b.add(&a_c).unwrap();

        assert_eq!(a_b_plus_c.data(), a_b_plus_a_c.data());
    }

    /// Test matrix multiplication with gradient computation
    #[test]
    fn test_matmul_gradients() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let result = a.matmul(&b).unwrap();
        assert!(result.requires_grad());
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test matrix multiplication numerical precision
    #[test]
    fn test_matmul_numerical_precision() {
        use approx::assert_relative_eq;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b).unwrap();

        // Verify mathematical correctness
        assert_relative_eq!(result.data()[0], 19.0, epsilon = 1e-15);
        assert_relative_eq!(result.data()[1], 22.0, epsilon = 1e-15);
        assert_relative_eq!(result.data()[2], 43.0, epsilon = 1e-15);
        assert_relative_eq!(result.data()[3], 50.0, epsilon = 1e-15);
    }

    /// Test matrix multiplication with large matrices
    #[test]
    fn test_matmul_large_matrices() {
        let size = 100;
        let mut data_a = vec![0.0; size * size];
        let mut data_b = vec![0.0; size * size];

        // Fill with some pattern
        for i in 0..size {
            for j in 0..size {
                data_a[i * size + j] = (i + j) as f64;
                data_b[i * size + j] = (i * j) as f64;
            }
        }

        let a = Tensor::from_vec(data_a, vec![size, size]);
        let b = Tensor::from_vec(data_b, vec![size, size]);
        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape(), vec![size, size]);

        // Verify some specific values
        assert_eq!(result.data()[0], 0.0); // First element should be 0
        assert_eq!(result.data()[size * size - 1], 0.0); // Last element should be 0
    }

    /// Test matrix multiplication with different data types
    #[test]
    fn test_matmul_different_types() {
        // Test with f32
        let a_f32 = Tensor::from_vec(vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32], vec![2, 2]);
        let b_f32 = Tensor::from_vec(vec![5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32], vec![2, 2]);
        let result_f32 = a_f32.matmul(&b_f32).unwrap();
        assert_eq!(result_f32.shape(), vec![2, 2]);

        // Test with integer types where applicable
        let a_i32 = Tensor::from_vec(vec![1_i32, 2_i32, 3_i32, 4_i32], vec![2, 2]);
        let b_i32 = Tensor::from_vec(vec![5_i32, 6_i32, 7_i32, 8_i32], vec![2, 2]);
        let result_i32 = a_i32.matmul(&b_i32).unwrap();
        assert_eq!(result_i32.shape(), vec![2, 2]);
    }

    /// Test matrix multiplication associativity
    #[test]
    fn test_matmul_associativity() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], vec![2, 2]);

        // (A * B) * C
        let ab = a.matmul(&b).unwrap();
        let abc = ab.matmul(&c).unwrap();

        // A * (B * C)
        let bc = b.matmul(&c).unwrap();
        let abc2 = a.matmul(&bc).unwrap();

        // Should be approximately equal
        for i in 0..4 {
            assert_relative_eq!(abc.data()[i], abc2.data()[i], epsilon = 1e-10);
        }
    }

    /// Test matrix multiplication with transpose
    #[test]
    fn test_matmul_with_transpose() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);

        // Test that A * B^T works correctly
        let b_t = b.transpose();
        let result = a.matmul(&b_t).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Manual calculation: [[1*7 + 2*10 + 3*8, 1*8 + 2*11 + 3*9], [4*7 + 5*10 + 6*8, 4*8 + 5*11 + 6*9]]
        assert_relative_eq!(result.data()[0], 50.0, epsilon = 1e-10);
        assert_relative_eq!(result.data()[1], 56.0, epsilon = 1e-10);
        assert_relative_eq!(result.data()[2], 113.0, epsilon = 1e-10);
        assert_relative_eq!(result.data()[3], 128.0, epsilon = 1e-10);
    }
}

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Perform matrix multiplication
    ///
    /// # Arguments
    /// * `other` - Right-hand side tensor for multiplication
    ///
    /// # Returns
    /// Result containing the matrix product or an error if shapes are incompatible
    ///
    /// # Errors
    /// Returns `TensorError::MatrixMulRequires2D` if either tensor is not at least 2D
    /// Returns `TensorError::IncompatibleMatrixDims` if matrix dimensions are incompatible
    /// Returns `TensorError::InvalidOperation` if batched matrix multiplication is attempted
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    /// let c = a.matmul(&b).unwrap();
    /// // c.shape() == [2, 2]
    /// ```
    pub fn matmul(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let _guard = crate::performance::global_context().start_measurement("matmul");

        // Validate dimensions
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err(crate::TensorError::MatrixMulRequires2D {
                lhs_shape: self.shape.clone(),
                rhs_shape: other.shape.clone(),
            });
        }

        let m = self.shape[self.shape.len() - 2];
        let k = self.shape[self.shape.len() - 1];
        let n = other.shape[other.shape.len() - 1];

        if k != other.shape[other.shape.len() - 2] {
            return Err(crate::TensorError::IncompatibleMatrixDims {
                lhs_m: m,
                lhs_k: k,
                rhs_k: other.shape[other.shape.len() - 2],
                rhs_n: n,
            });
        }

        // For simplicity, handle 2D case first
        if self.shape.len() == 2 && other.shape.len() == 2 {
            let mut result_data = vec![T::zero(); m * n];

            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        let a_val = self.data[i * k + l];
                        let b_val = other.data[l * n + j];
                        result_data[i * n + j] = result_data[i * n + j] + a_val * b_val;
                    }
                }
            }

            let mut result = Tensor::from_vec(result_data, vec![m, n]);

            // Create computational graph node if inputs require gradients
            if self.requires_grad() || other.requires_grad() {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let self_node = if let Some(node) = self.node {
                        node
                    } else {
                        let node = context.create_node(Operation::Add, vec![]);
                        let data_f64: Vec<f64> = self
                            .data
                            .iter()
                            .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                            .collect();
                        context.register_tensor(node, data_f64, self.shape.clone());
                        node
                    };

                    let other_node = if let Some(node) = other.node {
                        node
                    } else {
                        let node = context.create_node(Operation::Add, vec![]);
                        let data_f64: Vec<f64> = other
                            .data
                            .iter()
                            .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                            .collect();
                        context.register_tensor(node, data_f64, other.shape.clone());
                        node
                    };

                    let node_id = context.create_node(Operation::Matmul, vec![self_node, other_node]);
                    let result_data_f64: Vec<f64> = result
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, result_data_f64, result.shape.clone());
                    result.node = Some(node_id);
                });
            }

            Ok(result)
        } else {
            // Batched matrix multiplication not yet implemented
            Err(crate::TensorError::InvalidOperation {
                message: "Batched matrix multiplication not yet implemented".to_string(),
            })
        }
    }
}
