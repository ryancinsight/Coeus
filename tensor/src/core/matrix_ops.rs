//! Matrix operations for tensors
//!
//! This module contains matrix multiplication and other linear algebra operations
//! with automatic differentiation support.

use crate::{Tensor, TensorError, Dtype, FloatDtype, Result};
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
