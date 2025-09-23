//! Tensor arithmetic operations
//!
//! This module contains implementations of arithmetic operations for tensors,
//! including addition, subtraction, multiplication, and division with
//! broadcasting support and automatic differentiation integration.

use crate::TensorError;
use crate::{with_autograd_context, Dtype, Tensor};
use coeus_autograd::context::Operation;
use std::ops::{Add, Div, Mul, Sub};

/// Broadcast preparation result containing shape and data for both operands
struct BroadcastResult<T> {
    shape: Vec<usize>,
    lhs_data: Vec<T>,
    rhs_data: Vec<T>,
}

/// Helper function to perform broadcasting logic and return prepared data
/// This eliminates DRY violations across arithmetic operations
fn broadcast_and_prepare<T: Dtype + Copy>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<BroadcastResult<T>, TensorError> {
    if lhs.shape == rhs.shape {
        // Same shape - direct operation
        Ok(BroadcastResult {
            shape: lhs.shape.clone(),
            lhs_data: lhs.data.clone(),
            rhs_data: rhs.data.clone(),
        })
    } else if lhs.shape.is_empty() && !rhs.shape.is_empty() {
        // Broadcast lhs (scalar) to match rhs's shape
        let broadcast_value = lhs.data[0];
        let broadcast_data = vec![broadcast_value; rhs.numel()];
        Ok(BroadcastResult {
            shape: rhs.shape.clone(),
            lhs_data: broadcast_data,
            rhs_data: rhs.data.clone(),
        })
    } else if rhs.shape.is_empty() && !lhs.shape.is_empty() {
        // Broadcast rhs (scalar) to match lhs's shape
        let broadcast_value = rhs.data[0];
        let broadcast_data = vec![broadcast_value; lhs.numel()];
        Ok(BroadcastResult {
            shape: lhs.shape.clone(),
            lhs_data: lhs.data.clone(),
            rhs_data: broadcast_data,
        })
    } else {
        Err(TensorError::ShapeMismatch {
            expected: lhs.shape.clone(),
            actual: rhs.shape.clone(),
        })
    }
}

/// Helper function to create computational graph nodes for autograd
fn create_autograd_nodes<T: Dtype>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    operation: Operation,
) -> Option<u64> {
    if !lhs.requires_grad() && !rhs.requires_grad() {
        return None;
    }

    with_autograd_context(|context| {
        // Create leaf nodes for inputs that don't have them
        let lhs_node = if let Some(node) = lhs.node {
            node
        } else {
            let node = context.create_leaf_node();
            if let Some(data_f64) = lhs
                .data
                .iter()
                .map(|&x| Dtype::to_f64(&x))
                .collect::<Option<Vec<f64>>>()
            {
                context.register_tensor(node, data_f64, lhs.shape.clone());
            }
            node
        };

        let rhs_node = if let Some(node) = rhs.node {
            node
        } else {
            let node = context.create_leaf_node();
            if let Some(data_f64) = rhs
                .data
                .iter()
                .map(|&x| Dtype::to_f64(&x))
                .collect::<Option<Vec<f64>>>()
            {
                context.register_tensor(node, data_f64, rhs.shape.clone());
            }
            node
        };

        // Create operation node
        Some(context.create_node(operation, vec![lhs_node, rhs_node]))
    })
}

/// Helper function to create result tensor with proper autograd setup
fn create_result_tensor<T: Dtype>(
    data: Vec<T>,
    shape: Vec<usize>,
    device: crate::Device,
    layout: crate::Layout,
    autograd_node: Option<u64>,
) -> Tensor<T> {
    let mut result = Tensor {
        data,
        shape,
        device,
        layout,
        node: None,
        context: None,
        grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
        input_tensor_nodes: vec![],
        buffers: std::collections::HashMap::new(),
    };

    if let Some(node_id) = autograd_node {
        result.set_requires_grad(true);
        result.node = Some(node_id);

        // Register result tensor data for gradient computation (matches original behavior)
        with_autograd_context(|context| {
            let result_data_f64: Vec<f64> = result
                .data
                .iter()
                .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                .collect();
            context.register_tensor(node_id, result_data_f64, result.shape.clone());
        });
    }

    result
}

impl<T: Dtype + std::ops::Add<Output = T>> Add for &Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;

    fn add(self, other: Self) -> Self::Output {
        // Use helper function for broadcasting logic (eliminates DRY violations)
        let broadcast_result = broadcast_and_prepare(self, other)?;
        let result_shape = broadcast_result.shape;
        let lhs_data = broadcast_result.lhs_data;
        let rhs_data = broadcast_result.rhs_data;

        // Perform element-wise addition
        let result_data = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(a, b)| *a + *b)
            .collect();

        // Create autograd nodes if needed
        let autograd_node = create_autograd_nodes(self, other, Operation::Add);

        // Create result tensor using helper function
        let result = create_result_tensor(
            result_data,
            result_shape,
            self.device,
            self.layout,
            autograd_node,
        );

        Ok(result)
    }
}

impl<T: Dtype + std::ops::Sub<Output = T>> Sub for &Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;

    fn sub(self, other: Self) -> Self::Output {
        // Use helper function for broadcasting logic (eliminates DRY violations)
        let broadcast_result = broadcast_and_prepare(self, other)?;
        let result_shape = broadcast_result.shape;
        let lhs_data = broadcast_result.lhs_data;
        let rhs_data = broadcast_result.rhs_data;

        // Perform element-wise subtraction
        let result_data = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(a, b)| *a - *b)
            .collect();

        // Create autograd nodes if needed
        let autograd_node = create_autograd_nodes(self, other, Operation::Sub);

        // Create result tensor using helper function
        let result = create_result_tensor(
            result_data,
            result_shape,
            self.device,
            self.layout,
            autograd_node,
        );

        Ok(result)
    }
}

impl<T: Dtype + std::ops::Mul<Output = T>> Mul for &Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;

    fn mul(self, other: Self) -> Self::Output {
        // Use helper function for broadcasting logic (eliminates DRY violations)
        let broadcast_result = broadcast_and_prepare(self, other)?;
        let result_shape = broadcast_result.shape;
        let lhs_data = broadcast_result.lhs_data;
        let rhs_data = broadcast_result.rhs_data;

        // Perform element-wise multiplication
        let result_data = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(a, b)| *a * *b)
            .collect();

        // Create autograd nodes if needed
        let autograd_node = create_autograd_nodes(self, other, Operation::Mul);

        // Create result tensor using helper function
        let result = create_result_tensor(
            result_data,
            result_shape,
            self.device,
            self.layout,
            autograd_node,
        );

        Ok(result)
    }
}

impl<T: Dtype + std::ops::Div<Output = T>> Div for &Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;

    fn div(self, other: Self) -> Self::Output {
        // Use helper function for broadcasting logic (eliminates DRY violations)
        let broadcast_result = broadcast_and_prepare(self, other)?;
        let result_shape = broadcast_result.shape;
        let lhs_data = broadcast_result.lhs_data;
        let rhs_data = broadcast_result.rhs_data;

        // Perform element-wise division
        let result_data = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(a, b)| *a / *b)
            .collect();

        // Create autograd nodes if needed
        let autograd_node = create_autograd_nodes(self, other, Operation::Div);

        // Create result tensor using helper function
        let result = create_result_tensor(
            result_data,
            result_shape,
            self.device,
            self.layout,
            autograd_node,
        );

        Ok(result)
    }
}
