//! Shape manipulation operations for tensors
//!
//! This module contains operations for changing tensor shapes including
//! reshaping, squeezing, unsqueezing, and expanding.

use crate::{Tensor, TensorError, Dtype, Result};

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Reshape the tensor to a new shape
    ///
    /// # Arguments
    /// * `new_shape` - The desired new shape
    ///
    /// # Returns
    /// Result containing the reshaped tensor or an error if reshaping is invalid
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let reshaped = tensor.reshape(vec![4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[4]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: new_shape,
            });
        }

        // Create new tensor with same data but different shape
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            layout: self.layout,
            node: self.node,
            context: self.context.clone(),
            grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
            input_tensor_nodes: self.input_tensor_nodes.clone(),
        })
    }

    /// Add a dimension of size 1 at the specified position
    ///
    /// # Arguments
    /// * `dim` - Position where to insert the new dimension
    ///
    /// # Returns
    /// Result containing the tensor with unsqueezed dimension or an error if invalid
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let unsqueezed = tensor.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[1, 2]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor<T>> {
        if dim > self.shape.len() {
            return Err(TensorError::InvalidOperation {
                message: format!("Dimension {} out of bounds for {}D tensor", dim, self.shape.len())
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            layout: self.layout,
            node: self.node,
            context: self.context.clone(),
            grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
            input_tensor_nodes: self.input_tensor_nodes.clone(),
        })
    }

    /// Expand the tensor to a target shape by broadcasting
    ///
    /// # Arguments
    /// * `target_shape` - The desired shape to expand to
    ///
    /// # Returns
    /// Result containing the expanded tensor or an error if expansion is invalid
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let expanded = tensor.expand(vec![3, 2]).unwrap();
    /// // Broadcasting: [1.0, 2.0] -> [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    /// ```
    pub fn expand(&self, target_shape: Vec<usize>) -> Result<Tensor<T>> {
        // Basic broadcasting validation
        if target_shape.len() < self.shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: target_shape,
            });
        }

        // Check if shapes are compatible for broadcasting
        for (i, (&target_dim, &current_dim)) in target_shape.iter().rev().zip(self.shape.iter().rev()).enumerate() {
            if current_dim != 1 && current_dim != target_dim {
                return Err(TensorError::ShapeMismatch {
                    expected: self.shape.clone(),
                    actual: target_shape,
                });
            }
        }

        // Handle case where one dimension is 1
        if self.shape.len() == 1 && target_shape.len() == 2 &&
           self.shape[0] == target_shape[1] && target_shape[0] >= 1 {
            // Broadcast [N] to [M, N]
            let mut new_data = Vec::with_capacity(target_shape.iter().product());
            for _ in 0..target_shape[0] {
                new_data.extend_from_slice(&self.data);
            }
            Ok(Tensor::from_vec(new_data, target_shape))
        } else if self.shape.is_empty() && !target_shape.is_empty() {
            // Broadcast scalar to target shape
            let numel = target_shape.iter().product();
            let new_data = vec![self.data[0]; numel];
            Ok(Tensor::from_vec(new_data, target_shape))
        } else {
            // For more complex broadcasting, we'd need more sophisticated logic
            Err(TensorError::InvalidOperation {
                message: "Complex broadcasting not yet implemented".to_string()
            })
        }
    }
}
