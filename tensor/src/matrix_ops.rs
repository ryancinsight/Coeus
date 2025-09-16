//! Matrix operations for tensors
//!
//! This module contains matrix multiplication and other linear algebra operations
//! with automatic differentiation support.

use crate::{Tensor, Dtype};
use crate::with_autograd_context;
use coeus_autograd::context::Operation;

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
