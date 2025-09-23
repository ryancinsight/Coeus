//! Core trait for differentiable operations
//!
//! This module defines the Differentiable trait and related utilities
//! for implementing automatic differentiation operations.

use crate::tensor_ref::TensorRef;
use coeus_dtype::Dtype;

/// Core trait for differentiable operations
pub trait Differentiable<T: Dtype> {
    /// Compute the forward pass
    fn forward(&self, inputs: &[&TensorRef<T>]) -> TensorRef<T>;

    /// Compute the backward pass
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>>;
}

/// Context for tracking operations in the computational graph
#[derive(Clone)]
pub struct Context<T>
where
    T: coeus_dtype::FloatDtype + std::ops::Neg<Output = T>,
{
    graph: std::sync::Arc<parking_lot::RwLock<crate::graph::ComputationalGraph<T>>>,
    requires_grad: bool,
    next_node_id: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl<T> Context<T>
where
    T: coeus_dtype::FloatDtype + std::ops::Neg<Output = T>,
{
    /// Create a new context
    pub fn new(requires_grad: bool) -> Self {
        Self {
            graph: std::sync::Arc::new(parking_lot::RwLock::new(crate::graph::ComputationalGraph::new())),
            requires_grad,
            next_node_id: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Check if gradients are required
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the next node ID for this context
    pub fn next_node_id(&self) -> crate::graph::NodeId {
        use std::sync::atomic::Ordering;
        let id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        crate::graph::NodeId(id)
    }

    /// Get a reference to the computational graph
    pub fn graph(&self) -> &std::sync::Arc<parking_lot::RwLock<crate::graph::ComputationalGraph<T>>> {
        &self.graph
    }

    /// Enable gradient computation for this context
    pub fn requires_grad_mut(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
}

impl<T> Default for Context<T>
where
    T: coeus_dtype::FloatDtype + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new(false)
    }
}

/// Thread-safe gradient accumulator
#[derive(Clone)]
pub struct GradientAccumulator<T: coeus_dtype::Dtype> {
    gradients: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<crate::graph::NodeId, TensorRef<T>>>>,
}

impl<T: coeus_dtype::Dtype> GradientAccumulator<T> {
    /// Create a new gradient accumulator
    pub fn new() -> Self {
        Self {
            gradients: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Accumulate gradient for a node
    pub fn accumulate(&self, node_id: crate::graph::NodeId, gradient: TensorRef<T>) {
        let mut grads = self.gradients.write();
        if let Some(existing) = grads.get_mut(&node_id) {
            // Add gradients element-wise
            *existing = existing.add(&gradient);
        } else {
            grads.insert(node_id, gradient);
        }
    }

    /// Get gradient for a node
    pub fn get(&self, node_id: &crate::graph::NodeId) -> Option<TensorRef<T>> {
        self.gradients.read().get(node_id).cloned()
    }

    /// Clear all accumulated gradients
    pub fn clear(&self) {
        self.gradients.write().clear();
    }
}

impl<T: coeus_dtype::Dtype> Default for GradientAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Error types for automatic differentiation
#[derive(Debug, thiserror::Error)]
pub enum AutogradError {
    #[error("Computational graph error: {0}")]
    GraphError(String),

    #[error("Gradient computation error: {0}")]
    GradientError(String),

    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Cycle detected in computational graph")]
    CycleDetected,
}
