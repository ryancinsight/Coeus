//! Tensor operations module
//!
//! This module contains tensor operations that were extracted from the main tensor.rs
//! file to improve modularity and reduce file size.

use crate::{with_autograd_context, Result, TensorError};
use coeus_autograd::context::Operation;
use coeus_dtype::Dtype;
use std::ops::Add;

/// Additional tensor operations implementation
impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> crate::Tensor<T> {
    /// Reshape the tensor to a new shape
    ///
    /// # Arguments
    /// * `new_shape` - The new shape for the tensor
    ///
    /// # Returns
    /// A new tensor with the specified shape
    ///
    /// # Errors
    /// Returns an error if the new shape is incompatible with the tensor's size
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let reshaped = tensor.reshape(vec![4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[4]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: new_shape,
            });
        }

        let mut result = Self {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            layout: self.layout,
            node: None,
            context: None,
            grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
            input_tensor_nodes: vec![],
        };

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            with_autograd_context(|context| {
                let inputs = vec![self.node.unwrap()];
                let node_id = context.create_node(Operation::Reshape, inputs);

                if T::is_float() {
                    // Register tensor data for gradient computation
                    if let Some(data_f64) = self
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node_id, data_f64, self.shape.clone());
                    }
                }

                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Matrix multiplication (GEMM operation)
    ///
    /// Performs matrix multiplication: C = A @ B
    /// where A has shape (m, k) and B has shape (k, n), resulting in C with shape (m, n)
    ///
    /// # Arguments
    /// * `other` - Right-hand side tensor for matrix multiplication
    ///
    /// # Returns
    /// Result tensor of the matrix multiplication
    ///
    /// # Panics
    /// Panics if tensor dimensions are incompatible for matrix multiplication
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    pub fn matmul(&self, other: &crate::Tensor<T>) -> crate::Result<crate::Tensor<T>>
    where
        T: Add<Output = T> + std::ops::Mul<Output = T> + Copy,
    {
        // Check dimensions
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(crate::TensorError::InvalidOperation {
                message: format!("Both tensors must have at least 2 dimensions for matrix multiplication, got {:?}D and {:?}D", self_shape.len(), other_shape.len()),
            });
        }

        let m = self_shape[self_shape.len() - 2];
        let k = self_shape[self_shape.len() - 1];
        let n = other_shape[other_shape.len() - 1];

        if other_shape[other_shape.len() - 2] != k {
            return Err(crate::TensorError::InvalidOperation {
                message: format!("Incompatible shapes for matrix multiplication: {}x{} @ {}x{} (expected k={} to match other k={})",
                    m, k, other_shape[other_shape.len() - 2], n, k, other_shape[other_shape.len() - 2]),
            });
        }

        // Compute result shape
        let mut result_shape = Vec::new();
        if self_shape.len() > 2 {
            result_shape.extend_from_slice(&self_shape[..self_shape.len() - 2]);
        }
        if other_shape.len() > 2 {
            result_shape.extend_from_slice(&other_shape[..other_shape.len() - 2]);
        }
        result_shape.push(m);
        result_shape.push(n);

        // Perform matrix multiplication
        let mut result_data = vec![T::zero(); result_shape.iter().product()];

        // Simple implementation for now - can be optimized later
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    let self_idx = if self_shape.len() == 2 {
                        i * k + l
                    } else {
                        // Handle batched dimensions
                        let batch_idx = 0; // Simplified for now
                        batch_idx * (m * k) + i * k + l
                    };

                    let other_idx = if other_shape.len() == 2 {
                        l * n + j
                    } else {
                        // Handle batched dimensions
                        let batch_idx = 0; // Simplified for now
                        batch_idx * (k * n) + l * n + j
                    };

                    sum = sum + self.data[self_idx] * other.data[other_idx];
                }

                let result_idx = if result_shape.len() == 2 {
                    i * n + j
                } else {
                    // Handle batched dimensions
                    let batch_idx = 0; // Simplified for now
                    batch_idx * (m * n) + i * n + j
                };

                result_data[result_idx] = sum;
            }
        }

        let mut result = crate::Tensor::from_vec(result_data, result_shape);

        // Set gradient tracking and create computational graph node
        if self.requires_grad() || other.requires_grad() {
            result.set_requires_grad(true);

            // Create computational graph node if autograd is enabled
            with_autograd_context(|context| {
                // Ensure input nodes exist
                let self_node = if let Some(node) = self.node {
                    // Ensure tensor data is registered with this node if not already
                    if context.get_tensor_data(node).is_none() {
                        if let Some(data_f64) = self
                            .data
                            .iter()
                            .map(|&x| Dtype::to_f64(&x))
                            .collect::<Option<Vec<f64>>>()
                        {
                            context.register_tensor(node, data_f64, self.shape.clone());
                        }
                    }
                    node
                } else {
                    // Create leaf node without operation for tensors that don't have gradients yet
                    let node = context.create_leaf_node();
                    if let Some(data_f64) = self
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node, data_f64, self.shape.clone());
                    }
                    node
                };

                let other_node = if let Some(node) = other.node {
                    // Ensure tensor data is registered with this node if not already
                    if context.get_tensor_data(node).is_none() {
                        if let Some(data_f64) = other
                            .data
                            .iter()
                            .map(|&x| Dtype::to_f64(&x))
                            .collect::<Option<Vec<f64>>>()
                        {
                            context.register_tensor(node, data_f64, other.shape.clone());
                        }
                    }
                    node
                } else {
                    // Create leaf node without operation for tensors that don't have gradients yet
                    let node = context.create_leaf_node();
                    if let Some(data_f64) = other
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node, data_f64, other.shape.clone());
                    }
                    node
                };

                let node_id = context.create_node(Operation::Matmul, vec![self_node, other_node]);

                // Register result tensor data for gradient computation
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());

                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Unsqueeze a dimension at the specified position
    ///
    /// # Arguments
    /// * `dim` - Position to insert the new dimension
    ///
    /// # Returns
    /// New tensor with an additional dimension of size 1
    ///
    /// # Errors
    /// Returns an error if the dimension is out of bounds
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let unsqueezed = tensor.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<crate::Tensor<T>> {
        let mut new_shape = self.shape.clone();
        if dim > new_shape.len() {
            return Err(TensorError::InvalidDimension {
                dim,
                max_dim: new_shape.len(),
            });
        }
        new_shape.insert(dim, 1);

        let mut result = Self {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            layout: self.layout,
            node: None,
            context: None,
            grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
            input_tensor_nodes: vec![],
        };

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            with_autograd_context(|context| {
                let inputs = vec![self.node.unwrap()];
                let node_id = context.create_node(Operation::Unsqueeze, inputs);

                if T::is_float() {
                    // Register tensor data for gradient computation
                    if let Some(data_f64) = self
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node_id, data_f64, self.shape.clone());
                    }
                }

                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Expand tensor to a target shape using broadcasting
    ///
    /// # Arguments
    /// * `target_shape` - The shape to expand to
    ///
    /// # Returns
    /// New tensor expanded to the target shape
    ///
    /// # Errors
    /// Returns an error if shapes are incompatible for broadcasting
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let expanded = tensor.expand(vec![3, 2]).unwrap();
    /// assert_eq!(expanded.shape(), &[3, 2]);
    /// ```
    pub fn expand(&self, target_shape: Vec<usize>) -> Result<crate::Tensor<T>> {
        // Check if expansion is possible
        if self.shape.len() > target_shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: target_shape,
                actual: self.shape.clone(),
            });
        }

        // Prepend dimensions of size 1 as needed
        let mut expanded_shape = vec![1; target_shape.len() - self.shape.len()];
        expanded_shape.extend_from_slice(&self.shape);

        // Check broadcasting compatibility
        for (expanded, target) in expanded_shape.iter().zip(target_shape.iter()) {
            if *expanded != 1 && *expanded != *target {
                return Err(TensorError::ShapeMismatch {
                    expected: target_shape,
                    actual: self.shape.clone(),
                });
            }
        }

        // Create expanded data by repeating along broadcastable dimensions
        let mut result_data = Vec::new();
        let mut indices = vec![0; expanded_shape.len()];

        // Simple expansion implementation
        // This is a basic implementation - could be optimized
        for _ in 0..target_shape.iter().product::<usize>() {
            let mut source_idx = 0;
            let mut stride = 1;

            // Calculate source index from target indices
            for (_i, (&idx, &dim)) in indices.iter().zip(&expanded_shape).enumerate().rev() {
                if dim == 1 {
                    // Broadcast dimension - use 0
                    source_idx += 0;
                } else {
                    source_idx += idx * stride;
                }
                stride *= dim;
            }

            if source_idx < self.data.len() {
                result_data.push(self.data[source_idx]);
            } else {
                result_data.push(T::zero());
            }

            // Increment indices
            for i in (0..indices.len()).rev() {
                indices[i] += 1;
                if indices[i] < target_shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }

        let mut result = crate::Tensor::from_vec(result_data, target_shape);

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            with_autograd_context(|context| {
                // Ensure input node exists
                let self_node = if let Some(node) = self.node {
                    // Ensure tensor data is registered with this node if not already
                    if context.get_tensor_data(node).is_none() {
                        if let Some(data_f64) = self
                            .data
                            .iter()
                            .map(|&x| Dtype::to_f64(&x))
                            .collect::<Option<Vec<f64>>>()
                        {
                            context.register_tensor(node, data_f64, self.shape.clone());
                        }
                    }
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]); // Leaf node
                    if let Some(data_f64) = self
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node, data_f64, self.shape.clone());
                    }
                    node
                };

                let node_id = context.create_node(Operation::Expand, vec![self_node]);

                if T::is_float() {
                    // Register result tensor data for gradient computation
                    if let Some(result_data_f64) = result
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node_id, result_data_f64, result.shape.clone());
                    }
                }

                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Raise tensor elements to a power
    ///
    /// # Arguments
    /// * `exponent` - The exponent to raise each element to
    ///
    /// # Returns
    /// New tensor with elements raised to the specified power
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    /// let powered = tensor.pow(2.0);
    /// assert_eq!(powered.data(), &[4.0, 9.0]);
    /// ```
    pub fn pow(&self, exponent: T) -> crate::Tensor<T>
    where
        T: num_traits::Float,
    {
        let data = self.data.iter().map(|x| x.powf(exponent)).collect();

        let mut result = crate::Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            with_autograd_context(|context| {
                let inputs = vec![self.node.unwrap()];
                let node_id = context.create_node(
                    Operation::Pow(Dtype::to_f64(&exponent).unwrap_or(1.0)),
                    inputs,
                );

                if T::is_float() {
                    // Register tensor data for gradient computation
                    if let Some(data_f64) = self
                        .data
                        .iter()
                        .map(|&x| Dtype::to_f64(&x))
                        .collect::<Option<Vec<f64>>>()
                    {
                        context.register_tensor(node_id, data_f64, self.shape.clone());
                    }
                }

                result.node = Some(node_id);
            });
        }

        result
    }
}
