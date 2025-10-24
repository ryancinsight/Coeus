//! PyTorch-compatible automatic differentiation system
//!
//! This module provides gradient computation through automatic graph traversal,
//! compatible with `PyTorch`'s dynamic graph construction and backward pass.

use std::collections::HashSet;
use std::sync::Arc;

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageToDense};

use crate::error::{AutogradError, Result};

/// Gradient computation engine for automatic differentiation
///
/// Provides PyTorch-compatible backward pass through automatic graph traversal.
/// Unlike the abandoned node-based approach, this uses Function objects attached
/// to tensors via `grad_fn` for memory-efficient gradient computation.
#[derive(Debug, Default)]
pub struct GradientEngine {
    /// Set of visited functions during backward pass to prevent cycles
    visited: HashSet<usize>,
}

impl GradientEngine {
    /// Create a new gradient computation engine
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute gradients through automatic graph traversal
    ///
    /// This implements PyTorch-compatible backward pass by traversing the `grad_fn` chain.
    /// Unlike the abandoned node-based approach, this uses Function objects attached
    /// to tensors for memory-efficient gradient computation.
    ///
    /// # Arguments
    /// * `root_grad_fn` - The `grad_fn` of the tensor to start backward pass from
    /// * `grad_output` - Initial gradient w.r.t. the output tensor
    ///
    /// # Errors
    /// Returns error if backward pass fails
    pub fn backward<B, S, T>(
        &mut self,
        root_grad_fn: Option<&Arc<dyn coeus_tensor::Function<B, S, T>>>,
        grad_output: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<()>
    where
        B: Backend + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T> + core::fmt::Debug + Send + Sync + 'static + StorageToDense<T>,
        T: DataType,
    {
        if let Some(grad_fn) = root_grad_fn {
            self.backward_from_function(grad_fn, grad_output)?;
        }
        Ok(())
    }

    /// Recursive backward pass starting from a specific function
    fn backward_from_function<B, S, T>(
        &mut self,
        function: &Arc<dyn coeus_tensor::Function<B, S, T>>,
        grad_output: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<()>
    where
        B: Backend + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T> + core::fmt::Debug + Send + Sync + 'static + StorageToDense<T>,
        T: DataType,
    {
        // Prevent cycles by tracking visited functions
        let function_ptr = Arc::as_ptr(function);
        let function_id = function_ptr.cast::<()>() as usize;
        if !self.visited.insert(function_id) {
            return Ok(()); // Already processed this function
        }

        // Call function.backward() to compute gradients w.r.t. inputs
        let input_gradients =
            function
                .backward(grad_output)
                .map_err(|e| AutogradError::InvalidOperation {
                    operation: format!("Function backward failed: {e}"),
                })?;

        // Accumulate gradients into the input tensors
        let inputs = function.inputs();
        if inputs.len() != input_gradients.len() {
            return Err(AutogradError::InvalidInput {
                message: format!(
                    "Function {} returned {} gradients but has {} inputs",
                    function.name(),
                    input_gradients.len(),
                    inputs.len()
                ),
            });
        }

        // Accumulate gradients for each input tensor
        println!(
            "Accumulating {} gradients for {} inputs",
            input_gradients.len(),
            inputs.len()
        );
        for (i, (input_tensor_ref, grad_tensor)) in inputs.iter().zip(input_gradients).enumerate() {
            println!(
                "Accumulating gradient {}: input shape {:?}, grad shape {:?}",
                i,
                input_tensor_ref.shape().dims(),
                grad_tensor.shape().dims()
            );
            Self::accumulate_gradient(input_tensor_ref, grad_tensor)?;
        }

        // Recursively process parent functions
        // Each input tensor's gradient becomes the grad_output for its parent function
        for input_tensor_ref in inputs {
            if let Some(_parent_grad_fn) = input_tensor_ref.grad_fn() {
                // The gradient w.r.t. this input becomes the grad_output for the parent
                // For now, skip recursive processing until gradient accumulation is fixed
                // let input_grad = self.get_accumulated_gradient(input_tensor_ref)?;
                // self.backward_from_function(parent_grad_fn, &input_grad)?;
            }
        }

        Ok(())
    }

    /// Accumulate gradient into a tensor's grad field
    #[allow(clippy::used_underscore_binding)]
    fn accumulate_gradient<B, S, T>(
        tensor: &coeus_tensor::Tensor<B, S, T>,
        gradient: coeus_tensor::Tensor<B, S, T>,
    ) -> Result<()>
    where
        B: Backend + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T> + core::fmt::Debug + Send + Sync + Clone + 'static,
        T: DataType + Clone,
    {
        println!(
            "accumulate_gradient called for tensor with shape {:?}",
            tensor.shape().dims()
        );
        println!("gradient shape: {:?}", gradient.shape().dims());

        // For now, just set the gradient (no accumulation yet)
        // The gradient already has the correct storage type S
        let result = tensor.set_grad(gradient);
        println!("set_grad result: {result:?}");
        result.map_err(AutogradError::TensorError)
    }

    /// Get accumulated gradient for a tensor (always returns dense tensor)
    #[allow(dead_code)]
    fn get_accumulated_gradient<B, S, T>(
        tensor: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>>
    where
        B: Backend + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T> + core::fmt::Debug + Send + Sync + 'static + Clone,
        T: DataType,
    {
        tensor.grad().map_err(AutogradError::TensorError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_gradient_engine_creation() {
        let engine = GradientEngine::new();
        assert!(engine.visited.is_empty());
    }

    #[test]
    fn test_backward_with_none_grad_fn() {
        let mut engine = GradientEngine::new();
        let grad_tensor = coeus_tensor::Tensor::<
            coeus_backend::CpuBackend<Float32>,
            coeus_storage::DenseStorage<Float32>,
            Float32,
        >::from_vec(vec![Float32::new(1.0)], &[1])
        .unwrap();
        let result = engine.backward(None, &grad_tensor);
        assert!(result.is_ok());
    }
}
