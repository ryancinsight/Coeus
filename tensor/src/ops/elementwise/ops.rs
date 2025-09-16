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
pub fn add<T: crate::Dtype + std::ops::Add<Output = T>>(
    tensor: &Tensor<T>,
    other: &Tensor<T>,
) -> Result<Tensor<T>> {
    // Implement basic broadcasting support
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
        return Err(TensorError::ShapeMismatch {
            expected: tensor.shape.to_vec(),
            actual: other.shape.to_vec(),
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

            let node_id = context.create_node(Operation::Add, vec![tensor_node, other_node]);
            result.node = Some(node_id);

            // Store references to input tensors for gradient propagation
            result.input_tensor_nodes.push(Some(tensor_node));
            result.input_tensor_nodes.push(Some(other_node));
        });
    }

    Ok(result)
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
