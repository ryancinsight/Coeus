//! Custom autograd functions for user-defined differentiable operations.
//!
//! This module provides a simple API for users to define custom differentiable
//! operations that integrate with the automatic differentiation system.
//!
//! ## Example
//!
//! ```rust
//! use autograd::{custom::apply_custom_function, ops::backward};
//! use dtype::float::Float32;
//!
//! // Define a custom function that computes x^2
//! fn square_forward(inputs: &[&tensor::Tensor<backend::CpuBackend<Float32>, storage::DenseStorage<Float32>, Float32>])
//!     -> Result<tensor::Tensor<backend::CpuBackend<Float32>, storage::DenseStorage<Float32>, Float32>, Box<dyn std::error::Error>> {
//!     let input = &inputs[0];
//!     // Create a simple squared tensor (placeholder for actual computation)
//!     Ok((*input).clone()) // Simplified for doctest
//! }
//!
//! fn square_backward(grad_output: &tensor::Tensor<backend::CpuBackend<Float32>, storage::DenseStorage<Float32>, Float32>)
//!     -> anyhow::Result<Vec<tensor::Tensor<backend::CpuBackend<Float32>, storage::DenseStorage<Float32>, Float32>>> {
//!     // For f(x) = x², df/dx = 2*x, but simplified for demo
//!     Ok(vec![grad_output.clone()])
//! }
//!
//! // Use the custom function
//! let input = tensor::Tensor::from_vec(vec![Float32::new(3.0)], &[]).unwrap().requires_grad_(true);
//! let output = apply_custom_function(&[&input], square_forward, square_backward, "Square").unwrap();
//! backward(&output, None, false, false).unwrap();
//! ```

extern crate alloc;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;

use backend::Backend;
use dtype::DataType;

use storage::{DenseStorage, Storage};

use crate::{error::AutogradError, Result};

/// Apply a custom differentiable function with automatic differentiation setup.
///
/// This function allows users to define custom operations that integrate with
/// the autograd system, similar to `PyTorch`'s `Function.apply()`.
///
/// # Arguments
/// * `inputs` - Input tensors to the custom function
/// * `forward_fn` - Function that performs the forward pass
/// * `backward_fn` - Function that computes gradients for the backward pass
/// * `name` - Name of the custom function for debugging
///
/// # Returns
/// Output tensor with autograd setup
///
/// # Example
/// See module documentation for a complete example.
#[allow(clippy::missing_errors_doc)]
pub fn apply_custom_function<B, S, T>(
    inputs: &[&tensor::Tensor<B, S, T>],
    forward_fn: impl FnOnce(
        &[&tensor::Tensor<B, S, T>],
    )
        -> std::result::Result<tensor::Tensor<B, S, T>, Box<dyn std::error::Error>>,
    backward_fn: impl Fn(&tensor::Tensor<B, S, T>) -> anyhow::Result<Vec<tensor::Tensor<B, S, T>>>
        + Send
        + Sync
        + 'static,
    name: &'static str,
) -> Result<tensor::Tensor<B, S, T>>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T>
        + Clone
        + 'static
        + storage::StorageFromVec<T>
        + storage::StorageToDense<T>
        + core::fmt::Debug
        + Send
        + Sync,
    T: DataType,
{
    // Perform forward pass
    let output = forward_fn(inputs).map_err(|e| AutogradError::InvalidOperation {
        operation: alloc::format!("Forward pass failed: {e}"),
    })?;

    // Set up autograd if any input requires gradients
    let requires_grad = inputs.iter().any(|input| input.requires_grad());
    let output = if requires_grad {
        // Create a custom function wrapper
        let custom_fn = CustomFunction {
            backward_fn: Box::new(backward_fn),
            inputs: inputs
                .iter()
                .map(|input| Arc::new((**input).clone()))
                .collect(),
            name,
        };

        output.with_grad_fn(Some(Arc::new(custom_fn)))
    } else {
        output
    };

    Ok(output)
}

/// Internal wrapper for custom functions
pub(crate) struct CustomFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    #[allow(clippy::type_complexity)]
    backward_fn: Box<
        dyn Fn(&tensor::Tensor<B, S, T>) -> anyhow::Result<Vec<tensor::Tensor<B, S, T>>>
            + Send
            + Sync
            + 'static,
    >,
    inputs: Vec<crate::functions::TensorRef<B, S, T>>,
    name: &'static str,
}

impl<B, S, T> core::fmt::Debug for CustomFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CustomFunction({})", self.name)
    }
}

impl<B, S, T> tensor::AsAny for CustomFunction<B, S, T>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T> + Clone + core::fmt::Debug + Send + Sync + 'static,
    T: DataType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<B, S, T> tensor::DifferentiableFunction<B, S, T> for CustomFunction<B, S, T>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T> + Clone + core::fmt::Debug + Send + Sync + 'static,
    T: DataType,
{
    fn name(&self) -> &'static str {
        self.name
    }
}

impl<B, S, T> tensor::Function<B, S, T> for CustomFunction<B, S, T>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T>
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + storage::StorageFromVec<T>
        + storage::StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[crate::functions::TensorRef<B, S, T>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &tensor::Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<tensor::Tensor<B, S, T>>> {
        // Convert grad_output back to storage type S for the user's function
        let grad_data = grad_output.storage().as_slice().to_vec();
        let grad_dims = grad_output.shape().dims().to_vec();
        let grad_output_converted = tensor::Tensor::from_vec_with_backend(
            grad_data,
            &grad_dims,
            grad_output.backend().clone(),
        )?;

        (self.backward_fn)(&grad_output_converted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_custom_function() -> anyhow::Result<()> {
        // Define a simple function that doubles its input
        let forward_fn = |inputs: &[&tensor::Tensor<
            backend::CpuBackend<Float32>,
            storage::DenseStorage<Float32>,
            Float32,
        >]| {
            let input = &inputs[0];
            // Create a scalar tensor with value 2.0 for multiplication
            let two = tensor::Tensor::from_vec(vec![Float32::new(2.0)], &[])
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            Ok(&**input * &two)
        };

        let backward_fn = |grad_output: &tensor::Tensor<
            backend::CpuBackend<Float32>,
            storage::DenseStorage<Float32>,
            Float32,
        >| {
            // Gradient is always 2.0 for f(x) = 2*x
            let two = tensor::Tensor::from_vec(vec![Float32::new(2.0)], &[])?;
            let grad_input = grad_output * &two;
            Ok(vec![grad_input])
        };

        // Create input tensor
        let input = tensor::Tensor::from_vec(vec![Float32::new(3.0)], &[])?.requires_grad_(true);

        // Apply custom function
        let output = apply_custom_function(&[&input], forward_fn, backward_fn, "Double")?;

        // Check forward pass result
        assert_eq!(output.shape().dims(), &[]);
        assert!(output.function_object().is_some());

        // Test backward pass
        crate::ops::backward(&output, None, false, false)?;

        Ok(())
    }
}
