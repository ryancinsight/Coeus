//! Operation implementation for computational graph
//!
//! This module contains the Operation struct and related functionality
//! for representing operations in the computational graph.

use crate::{Dtype, NodeId, TensorRef};
use std::sync::Arc;

/// Represents an operation in the computational graph
#[derive(Clone)]
pub struct Operation<T: Dtype> {
    /// Operation name for debugging
    pub name: String,
    /// Input node IDs
    pub inputs: Vec<NodeId>,
    /// Output node ID
    pub output: NodeId,
    /// Backward operation function
    #[allow(clippy::type_complexity)]
    pub backward_fn:
        Arc<dyn Fn(&[&TensorRef<T>], &TensorRef<T>) -> Vec<TensorRef<T>> + Send + Sync>,
    /// Optional second-order derivative function for higher-order autodiff
    #[allow(clippy::type_complexity)]
    pub hessian_fn: Option<
        Arc<
            dyn Fn(&[&TensorRef<T>], &TensorRef<T>, &[&TensorRef<T>]) -> Vec<Vec<TensorRef<T>>>
                + Send
                + Sync,
        >,
    >,
}

impl<T: Dtype> Operation<T> {
    /// Create a new operation
    pub fn new<F>(
        name: impl Into<String>,
        inputs: Vec<NodeId>,
        output: NodeId,
        backward_fn: F,
    ) -> Self
    where
        F: Fn(&[&TensorRef<T>], &TensorRef<T>) -> Vec<TensorRef<T>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            inputs,
            output,
            backward_fn: Arc::new(backward_fn),
            hessian_fn: None,
        }
    }

    /// Execute the backward pass
    pub fn backward(
        &self,
        inputs: &[&TensorRef<T>],
        output_grad: &TensorRef<T>,
    ) -> Vec<TensorRef<T>> {
        (self.backward_fn)(inputs, output_grad)
    }

    /// Create a new operation with Hessian support
    pub fn with_hessian<F, H>(
        name: impl Into<String>,
        inputs: Vec<NodeId>,
        output: NodeId,
        backward_fn: F,
        hessian_fn: H,
    ) -> Self
    where
        F: Fn(&[&TensorRef<T>], &TensorRef<T>) -> Vec<TensorRef<T>> + Send + Sync + 'static,
        H: Fn(&[&TensorRef<T>], &TensorRef<T>, &[&TensorRef<T>]) -> Vec<Vec<TensorRef<T>>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            inputs,
            output,
            backward_fn: Arc::new(backward_fn),
            hessian_fn: Some(Arc::new(hessian_fn)),
        }
    }

    /// Execute the Hessian computation
    pub fn hessian(
        &self,
        inputs: &[&TensorRef<T>],
        output_grad: &TensorRef<T>,
        input_grads: &[&TensorRef<T>],
    ) -> Option<Vec<Vec<TensorRef<T>>>> {
        self.hessian_fn
            .as_ref()
            .map(|f| f(inputs, output_grad, input_grads))
    }
}
