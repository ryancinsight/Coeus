//! Element-wise arithmetic operations implementation
//!
//! This module contains the implementation of element-wise operations
//! with broadcasting support.

use crate::{Result, Tensor, TensorError};
use coeus_autograd::ContextOperation as Operation;
use rayon::prelude::*;

/// Element-wise addition of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise sum or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Add};
///
/// let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
/// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
///
/// let result = a.add(&b).unwrap();
/// assert_eq!(result.data(), &[4.0, 6.0]);
/// ```
/// Compute broadcast shape following NumPy/PyTorch broadcasting rules
fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);

    let mut result_shape = Vec::with_capacity(max_len);

    // Pad shorter shape with leading dimensions of size 1
    let padded_shape1 = if len1 < max_len {
        let padding = vec![1; max_len - len1];
        [padding.as_slice(), shape1].concat()
    } else {
        shape1.to_vec()
    };

    let padded_shape2 = if len2 < max_len {
        let padding = vec![1; max_len - len2];
        [padding.as_slice(), shape2].concat()
    } else {
        shape2.to_vec()
    };

    // Compute broadcast shape
    for (dim1, dim2) in padded_shape1.iter().zip(padded_shape2.iter()) {
        if *dim1 == *dim2 {
            result_shape.push(*dim1);
        } else if *dim1 == 1 {
            result_shape.push(*dim2);
        } else if *dim2 == 1 {
            result_shape.push(*dim1);
        } else {
            return Err(crate::TensorError::BroadcastingError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }
    }

    Ok(result_shape)
}

/// Broadcast tensor data to match target shape
fn broadcast_data<T: crate::Dtype + Clone + Copy>(
    data: &[T],
    original_shape: &[usize],
    target_shape: &[usize],
) -> Vec<T> {
    let original_numel = original_shape.iter().product::<usize>();
    let target_numel = target_shape.iter().product::<usize>();

    if original_numel == target_numel {
        // Same total size - just clone
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(target_numel);

    // Compute strides for original shape
    let mut original_strides = vec![1; original_shape.len()];
    for i in (0..original_shape.len() - 1).rev() {
        original_strides[i] = original_strides[i + 1] * original_shape[i + 1];
    }

    // Compute strides for target shape
    let mut target_strides = vec![1; target_shape.len()];
    for i in (0..target_shape.len() - 1).rev() {
        target_strides[i] = target_strides[i + 1] * target_shape[i + 1];
    }

    // Broadcast data
    for target_idx in 0..target_numel {
        let mut original_idx = 0;

        for (dim, (&orig_stride, &target_stride)) in original_strides
            .iter()
            .zip(&target_strides)
            .enumerate()
            .rev()
        {
            let target_coord =
                (target_idx / target_stride) % target_shape[target_shape.len() - 1 - dim];
            let orig_coord = if original_shape[original_shape.len() - 1 - dim] == 1 {
                0
            } else {
                target_coord
            };
            original_idx += orig_coord * orig_stride;
        }

        if original_idx < data.len() {
            result.push(data[original_idx]);
        } else {
            // Fallback for edge cases
            result.push(data[0]);
        }
    }

    result
}

pub fn add<T: crate::Dtype + std::ops::Add<Output = T>>(
    tensor: &Tensor<T>,
    other: &Tensor<T>,
) -> Result<Tensor<T>> {
    // Try advanced broadcasting first
    match compute_broadcast_shape(&tensor.shape, &other.shape) {
        Ok(result_shape) => {
            // Broadcast both tensors to result shape
            let tensor_data = if tensor.shape == result_shape {
                tensor.data.clone()
            } else {
                broadcast_data(&tensor.data, &tensor.shape, &result_shape)
            };

            let other_data = if other.shape == result_shape {
                other.data.clone()
            } else {
                broadcast_data(&other.data, &other.shape, &result_shape)
            };

            // Use parallel processing for large tensors to improve performance
            let data: Vec<T> = if tensor_data.len() > 1000 {
                tensor_data
                    .par_iter()
                    .zip(&other_data)
                    .map(|(a, b)| *a + *b)
                    .collect()
            } else {
                tensor_data
                    .iter()
                    .zip(&other_data)
                    .map(|(a, b)| *a + *b)
                    .collect()
            };

            let mut result = Tensor {
                data,
                shape: result_shape,
                device: tensor.device,
                layout: tensor.layout,
                node: None,
                context: None,
                grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
                input_tensor_nodes: vec![],
                buffers: std::collections::HashMap::new(),
            };

            // Set requires_grad if either input requires gradients
            if tensor.requires_grad() || other.requires_grad() {
                result.node = Some(0); // Will be set properly by autograd context
            }

            Ok(result)
        }
        Err(_) => {
            // Fallback to original simple broadcasting for backward compatibility
            let (result_shape, tensor_data, other_data) = if tensor.shape == other.shape {
                // Same shape - direct addition
                (
                    tensor.shape.clone(),
                    tensor.data.clone(),
                    other.data.clone(),
                )
            } else if tensor.shape.is_empty() && !other.shape.is_empty() {
                // Broadcast scalar tensor to match other's shape
                let broadcast_value = tensor.data[0];
                let broadcast_data = vec![broadcast_value; other.numel()];
                (other.shape.clone(), broadcast_data, other.data.clone())
            } else if other.shape.is_empty() && !tensor.shape.is_empty() {
                // Broadcast scalar other to match tensor's shape
                let broadcast_value = other.data[0];
                let broadcast_data = vec![broadcast_value; tensor.numel()];
                (tensor.shape.clone(), tensor.data.clone(), broadcast_data)
            } else {
                return Err(crate::TensorError::BroadcastingError {
                    shape1: tensor.shape.to_vec(),
                    shape2: other.shape.to_vec(),
                });
            };

            // Use parallel processing for large tensors to improve performance
            let data: Vec<T> = if tensor_data.len() > 1000 {
                tensor_data
                    .par_iter()
                    .zip(&other_data)
                    .map(|(a, b)| *a + *b)
                    .collect()
            } else {
                tensor_data
                    .iter()
                    .zip(&other_data)
                    .map(|(a, b)| *a + *b)
                    .collect()
            };

            let mut result = Tensor {
                data,
                shape: result_shape,
                device: tensor.device,
                layout: tensor.layout,
                node: None,
                context: None,
                grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
                input_tensor_nodes: vec![],
                buffers: std::collections::HashMap::new(),
            };

            // Set requires_grad if either input requires gradients
            if tensor.requires_grad() || other.requires_grad() {
                result.node = Some(0); // Will be set properly by autograd context
            }

            Ok(result)
        }
    }
}

/// Element-wise subtraction of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise difference or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Sub};
///
/// let a = Tensor::from_vec(vec![5.0, 7.0], vec![2]);
/// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
///
/// let result = a.sub(&b).unwrap();
/// assert_eq!(result.data(), &[2.0, 3.0]);
/// ```
pub fn sub<T: crate::Dtype + std::ops::Sub<Output = T> + crate::Dtype>(
    tensor: &Tensor<T>,
    other: &Tensor<T>,
) -> Result<Tensor<T>> {
    // For now, implement simple same-shape subtraction
    // Broadcasting logic will be implemented separately
    if tensor.shape != other.shape {
        return Err(TensorError::ShapeMismatch {
            expected: tensor.shape.to_vec(),
            actual: other.shape.to_vec(),
        });
    }

    let data = tensor
        .data
        .iter()
        .zip(&other.data)
        .map(|(a, b)| *a - *b)
        .collect();

    let mut result = Tensor {
        data,
        shape: tensor.shape.clone(),
        device: tensor.device,
        layout: tensor.layout,
        node: None,
        context: None,
        grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
        input_tensor_nodes: vec![],
        buffers: std::collections::HashMap::new(),
    };

    // Create computational graph node if either input requires gradients
    if tensor.requires_grad() || other.requires_grad() {
        result.set_requires_grad(true);
        crate::with_autograd_context(|context| {
            // Ensure input nodes exist
            let tensor_node = if let Some(node) = tensor.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, tensor.shape.clone());
                node
            };

            let other_node = if let Some(node) = other.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = other
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, other.shape.clone());
                node
            };

            let node_id = context.create_node(Operation::Sub, vec![tensor_node, other_node]);
            result.node = Some(node_id);

            // Store references to input tensors for gradient propagation
            result.input_tensor_nodes.push(Some(tensor_node));
            result.input_tensor_nodes.push(Some(other_node));
        });
    }

    Ok(result)
}

/// Element-wise multiplication of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise product or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Mul};
///
/// let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
/// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);
///
/// let result = a.mul(b).unwrap();
/// assert_eq!(result.data(), &[8.0, 15.0]);
/// ```
pub fn mul<T: crate::Dtype + std::ops::Mul<Output = T> + crate::Dtype>(
    tensor: &Tensor<T>,
    other: &Tensor<T>,
) -> Result<Tensor<T>> {
    // For now, implement simple same-shape multiplication
    // Broadcasting logic will be implemented separately
    if tensor.shape != other.shape {
        return Err(TensorError::ShapeMismatch {
            expected: tensor.shape.to_vec(),
            actual: other.shape.to_vec(),
        });
    }

    let data = tensor
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| *a * *b)
        .collect::<Vec<T>>();

    let mut result = Tensor {
        data,
        shape: tensor.shape.clone(),
        device: tensor.device,
        layout: tensor.layout,
        node: None,
        context: None,
        grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
        input_tensor_nodes: vec![],
        buffers: std::collections::HashMap::new(),
    };

    // Create computational graph node if either input requires gradients
    if tensor.requires_grad() || other.requires_grad() {
        result.set_requires_grad(true);
        crate::with_autograd_context(|context| {
            // Ensure input nodes exist
            let tensor_node = if let Some(node) = tensor.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, tensor.shape.clone());
                node
            };

            let other_node = if let Some(node) = other.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = other
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, other.shape.clone());
                node
            };

            let node_id = context.create_node(Operation::Mul, vec![tensor_node, other_node]);
            result.node = Some(node_id);

            // Store references to input tensors for gradient propagation
            result.input_tensor_nodes.push(Some(tensor_node));
            result.input_tensor_nodes.push(Some(other_node));
        });
    }

    Ok(result)
}

/// Element-wise division of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise quotient or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Div};
///
/// let a = Tensor::from_vec(vec![8.0, 15.0], vec![2]);
/// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);
///
/// let result = a.div(&b).unwrap();
/// assert_eq!(result.data(), &[2.0, 3.0]);
/// ```
pub fn div<T: crate::Dtype + std::ops::Div<Output = T> + crate::Dtype>(
    tensor: &Tensor<T>,
    other: &Tensor<T>,
) -> Result<Tensor<T>> {
    // For now, implement simple same-shape division
    // Broadcasting logic will be implemented separately
    if tensor.shape != other.shape {
        return Err(TensorError::ShapeMismatch {
            expected: tensor.shape.to_vec(),
            actual: other.shape.to_vec(),
        });
    }

    let data = tensor
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| *a / *b)
        .collect::<Vec<T>>();

    let mut result = Tensor {
        data,
        shape: tensor.shape.clone(),
        device: tensor.device,
        layout: tensor.layout,
        node: None,
        context: None,
        grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
        input_tensor_nodes: vec![],
        buffers: std::collections::HashMap::new(),
    };

    // Create computational graph node if either input requires gradients
    if tensor.requires_grad() || other.requires_grad() {
        result.set_requires_grad(true);
        crate::with_autograd_context(|context| {
            // Ensure input nodes exist
            let tensor_node = if let Some(node) = tensor.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, tensor.shape.clone());
                node
            };

            let other_node = if let Some(node) = other.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = other
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, other.shape.clone());
                node
            };

            let node_id = context.create_node(Operation::Div, vec![tensor_node, other_node]);
            result.node = Some(node_id);

            // Store references to input tensors for gradient propagation
            result.input_tensor_nodes.push(Some(tensor_node));
            result.input_tensor_nodes.push(Some(other_node));
        });
    }

    Ok(result)
}

/// Element-wise negation of a tensor
///
/// # Arguments
/// * `tensor` - Tensor to negate (mutable for potential gradient tracking)
///
/// # Returns
/// The element-wise negation of the tensor
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Neg};
///
/// let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
/// let result = a.neg();
/// assert_eq!(result.data(), &[-1.0, 2.0, -3.0]);
/// ```
pub fn neg<T: crate::Dtype + std::ops::Neg<Output = T> + crate::Dtype>(
    tensor: &Tensor<T>,
) -> Tensor<T> {
    let data = tensor.data.iter().map(|x| -*x).collect();

    let mut result = Tensor {
        data,
        shape: tensor.shape.clone(),
        device: tensor.device,
        layout: tensor.layout,
        node: None,
        context: None,
        grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
        input_tensor_nodes: vec![],
        buffers: std::collections::HashMap::new(),
    };

    // Create computational graph node if input requires gradients
    if tensor.requires_grad() {
        result.set_requires_grad(true);
        crate::with_autograd_context(|context| {
            // Ensure input node exists
            let tensor_node = if let Some(node) = tensor.node {
                node
            } else {
                let node = context.create_node(Operation::Add, vec![]);
                // Convert tensor data to f64 for gradient computation
                let data_f64: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| crate::Dtype::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node, data_f64, tensor.shape.clone());
                node
            };

            let node_id = context.create_node(Operation::Neg, vec![tensor_node]);
            result.node = Some(node_id);

            // Store reference to input tensor for gradient propagation
            result.input_tensor_nodes.push(Some(tensor_node));
        });
    }

    result
}

// Operator overload implementations moved to core/tensor.rs to avoid conflicts

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let result = add(&a, &b).unwrap();
        assert_eq!(result.data, &[4.0, 6.0]);
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::from_vec(vec![5.0, 7.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let result = sub(&a, &b).unwrap();
        assert_eq!(result.data, &[2.0, 3.0]);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);

        let result = mul(&a, &b).unwrap();
        assert_eq!(result.data, &[8.0, 15.0]);
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::from_vec(vec![8.0, 15.0], vec![2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);

        let result = div(&a, &b).unwrap();
        assert_eq!(result.data, &[2.0, 3.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
        let result = neg(&a);
        assert_eq!(result.data, &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_operator_overloads() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        // Test addition
        let result = (&a + &b).unwrap();
        assert_eq!(result.data, &[4.0, 6.0]);

        // Test subtraction
        let result = (&a - &b).unwrap();
        assert_eq!(result.data, &[-2.0, -2.0]);

        // Test multiplication
        let result = (&a * &b).unwrap();
        assert_eq!(result.data, &[3.0, 8.0]);

        // Test division
        let result = (&b / &a).unwrap();
        assert_eq!(result.data, &[3.0, 2.0]);

        // Test negation
        let result = -&a;
        assert_eq!(result.data, &[-1.0, -2.0]);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);

        // Test addition with different shapes
        let result = &a + &b;
        assert!(result.is_err());

        // Test subtraction with different shapes
        let result = &a - &b;
        assert!(result.is_err());

        // Test multiplication with different shapes
        let result = &a * &b;
        assert!(result.is_err());

        // Test division with different shapes
        let result = &a / &b;
        assert!(result.is_err());
    }
}
