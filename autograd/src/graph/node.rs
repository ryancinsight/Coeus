//! Node implementation for computational graph
//!
//! This module contains the Node struct and related functionality
//! for representing nodes in the computational graph.

use crate::{AutogradError, Dtype, Operation, TensorRef};
use std::fmt;

/// Node in the computational graph
#[derive(Clone)]
pub struct Node<T: Dtype> {
    /// Unique identifier for this node
    pub id: NodeId,
    /// The tensor data this node represents
    pub data: TensorRef<T>,
    /// Gradient with respect to this node
    pub grad: Option<TensorRef<T>>,
    /// Operation that produced this node (None for leaf nodes)
    pub operation: Option<Operation<T>>,
    /// Reference count for memory management
    pub ref_count: usize,
    /// Whether this node requires gradients
    pub requires_grad: bool,
}

impl<T: Dtype> Node<T> {
    /// Create a new node
    pub fn new(
        id: NodeId,
        data: TensorRef<T>,
        operation: Option<Operation<T>>,
        requires_grad: bool,
    ) -> Self {
        Self {
            id,
            data,
            grad: None,
            operation,
            ref_count: 1,
            requires_grad,
        }
    }

    /// Create a leaf node (no operation, input tensor)
    pub fn leaf(id: NodeId, data: TensorRef<T>, requires_grad: bool) -> Self {
        Self::new(id, data, None, requires_grad)
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.operation.is_none()
    }

    /// Set the gradient for this node
    pub fn set_grad(&mut self, grad: TensorRef<T>) {
        self.grad = Some(grad);
    }

    /// Accumulate gradient (add to existing gradient if any)
    pub fn accumulate_grad(&mut self, grad: &TensorRef<T>) -> Result<(), AutogradError> {
        if let Some(existing_grad) = &mut self.grad {
            // Add gradients element-wise
            let new_grad_data = existing_grad
                .data()
                .iter()
                .zip(grad.data().iter())
                .map(|(&a, &b)| a + b)
                .collect::<Vec<_>>();
            let new_grad = TensorRef::from_data(new_grad_data, grad.shape().to_vec());
            *existing_grad = new_grad;
        } else {
            self.grad = Some(grad.clone());
        }
        Ok(())
    }

    /// Increment reference count
    pub fn increment_ref_count(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn decrement_ref_count(&mut self) -> usize {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count
    }

    /// Get the operation that produced this node
    pub fn operation(&self) -> Option<&Operation<T>> {
        self.operation.as_ref()
    }
}

impl<T: Dtype> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("data_shape", &self.data.shape())
            .field("has_grad", &self.grad.is_some())
            .field("is_leaf", &self.is_leaf())
            .field("ref_count", &self.ref_count)
            .finish()
    }
}

/// Unique identifier for nodes in the computational graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}
