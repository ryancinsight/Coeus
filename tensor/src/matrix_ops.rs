//! Matrix operations for tensors
//!
//! This module contains matrix multiplication and other linear algebra operations
//! with automatic differentiation support.

use crate::{Tensor, Dtype};
use crate::with_autograd_context;
use coeus_autograd::context::Operation;

#[cfg(test)]
mod matrix_ops_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test basic 2D matrix multiplication
    #[test]
    fn test_matmul_2d_basic() {
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

    /// Test helper function broadcast_shapes
    #[test]
    fn test_broadcast_shapes() {
        // Test compatible shapes
        let a = vec![2, 3];
        let b = vec![3];
        let result = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(result, &[2, 3]);

        // Test incompatible shapes
        let c = vec![2, 3];
        let d = vec![4, 3];
        let result_err = broadcast_shapes(&c, &d);
        assert!(result_err.is_err());
    }

    /// Test helper function unravel_index
    #[test]
    fn test_unravel_index() {
        let shape = vec![2, 3, 4];
        let flat_index = 5;
        let coords = unravel_index(flat_index, &shape);
        assert_eq!(coords, &[0, 1, 1]); // 5 = 0*12 + 1*4 + 1*1
    }

    /// Test batched matrix multiplication
    #[test]
    fn test_matmul_batched() {
        // Test 3D batched multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);

        // Verify first batch: [[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
        assert_eq!(result.data()[3], 4.0);

        // Verify second batch: [[5, 6], [7, 8]] * [[2, 0], [0, 2]] = [[10, 12], [14, 16]]
        assert_eq!(result.data()[4], 10.0);
        assert_eq!(result.data()[5], 12.0);
        assert_eq!(result.data()[6], 14.0);
        assert_eq!(result.data()[7], 16.0);
    }

    /// Test batched matrix multiplication with broadcasting
    #[test]
    fn test_matmul_batched_broadcasting() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]); // Single batch
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]); // Two batches
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);

        // Both batches should have the same result since a is broadcasted
        for i in 0..4 {
            assert_eq!(result.data()[i], result.data()[i + 4]);
        }
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
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    /// let c = a.matmul(&b).unwrap();
    /// // c.shape() == [2, 2]
    /// ```
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, crate::Error> {
        // Validate dimensions
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err(crate::Error::InvalidShape {
                expected: "at least 2D".to_string(),
                got: format!("{:?}D and {:?}D", self.shape.len(), other.shape.len()),
            });
        }

        let m = self.shape[self.shape.len() - 2];
        let k = self.shape[self.shape.len() - 1];
        let n = other.shape[other.shape.len() - 1];

        if k != other.shape[other.shape.len() - 2] {
            return Err(crate::Error::InvalidShape {
                expected: format!("{}x{} compatible with {}x{}", m, k, other.shape[other.shape.len() - 2], n),
                got: format!("{}x{} vs {}x{}", m, k, other.shape[other.shape.len() - 2], n),
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

            result
        } else {
            // Handle batched matrix multiplication for higher-dimensional tensors
            self.matmul_batched(other)?
        }
    }

    /// Perform batched matrix multiplication for higher-dimensional tensors
    fn matmul_batched(&self, other: &Tensor<T>) -> Result<Tensor<T>, crate::Error> {
        // Validate dimensions
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err(crate::Error::InvalidShape {
                expected: "at least 2D".to_string(),
                got: format!("{:?}D and {:?}D", self.shape.len(), other.shape.len()),
            });
        }

        // For batched operations, the batch dimensions must be broadcastable
        let self_batch_dims = &self.shape[..self.shape.len() - 2];
        let other_batch_dims = &other.shape[..other.shape.len() - 2];

        // Broadcast batch dimensions
        let batch_shape = broadcast_shapes(self_batch_dims, other_batch_dims)?;

        let m = self.shape[self.shape.len() - 2];
        let k = self.shape[self.shape.len() - 1];
        let n = other.shape[other.shape.len() - 1];

        if k != other.shape[other.shape.len() - 2] {
            return Err(crate::Error::InvalidShape {
                expected: format!("{}x{} compatible with {}x{}", m, k, other.shape[other.shape.len() - 2], n),
                got: format!("{}x{} vs {}x{}", m, k, other.shape[other.shape.len() - 2], n),
            });
        }

        // Calculate total number of matrix multiplications needed
        let total_batch_size: usize = batch_shape.iter().product();
        let matrix_size = m * n;

        // Create result tensor
        let mut result_shape = batch_shape.clone();
        result_shape.push(m);
        result_shape.push(n);

        let mut result_data = vec![T::zero(); total_batch_size * matrix_size];

        // Perform batched matrix multiplication
        for batch_idx in 0..total_batch_size {
            let batch_coords = unravel_index(batch_idx, &batch_shape);

            // Get the matrix data for this batch
            let self_offset = batch_idx * (m * k);
            let other_offset = batch_idx * (k * n);
            let result_offset = batch_idx * matrix_size;

            // Perform matrix multiplication for this batch element
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        let a_idx = self_offset + i * k + l;
                        let b_idx = other_offset + l * n + j;
                        sum = sum + self.data[a_idx] * other.data[b_idx];
                    }
                    result_data[result_offset + i * n + j] = sum;
                }
            }
        }

        let mut result = Tensor::from_vec_raw(result_data, result_shape).map_err(|e| crate::Error::InvalidShape {
            expected: format!("valid tensor shape {:?}", result_shape),
            got: e.to_string(),
        })?;

        // Handle autograd if needed
        if self.requires_grad || other.requires_grad {
            with_autograd_context(|context| {
                let node_id = context.next_node_id();

                // Create backward operation for batched matmul
                let backward_op = Operation::MatMulBackward {
                    left_shape: self.shape.clone(),
                    right_shape: other.shape.clone(),
                };

                context.register_operation(node_id, backward_op);

                // Register input-output relationships
                if let Some(self_node) = self.node {
                    context.register_dependency(self_node, node_id);
                }
                if let Some(other_node) = other.node {
                    context.register_dependency(other_node, node_id);
                }

                // Convert result data to f64 for autograd
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();

                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }
}

/// Helper function to broadcast two shapes for batched operations
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, crate::Error> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = a.get(a.len().wrapping_sub(i + 1)).copied().unwrap_or(1);
        let b_dim = b.get(b.len().wrapping_sub(i + 1)).copied().unwrap_or(1);

        if a_dim == 1 && b_dim != 1 {
            result.push(b_dim);
        } else if a_dim != 1 && b_dim == 1 {
            result.push(a_dim);
        } else if a_dim == b_dim {
            result.push(a_dim);
        } else {
            return Err(crate::Error::InvalidShape {
                expected: format!("broadcastable dimensions"),
                got: format!("cannot broadcast {} and {}", a_dim, b_dim),
            });
        }
    }

    result.reverse();
    Ok(result)
}

/// Helper function to convert flat index to multi-dimensional coordinates
fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());

    for &dim in shape.iter().rev() {
        coords.push(flat_index % dim);
        flat_index /= dim;
    }

    coords.reverse();
    coords
}
