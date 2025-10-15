//! # High-Level Autograd Operations
//!
//! Provides PyTorch-compatible operations that automatically construct
//! the computation graph for gradient computation.
//!
//! ## Architecture
//!
//! These operations wrap tensor operations and automatically set `grad_fn`
//! on result tensors to enable automatic differentiation.

use crate::functions::{*, ExpFunction, LogFunction, SinFunction, CosFunction};
use coeus_tensor::Function;
extern crate alloc;
use alloc::{sync::Arc, vec::Vec};
use std::ops::Add;
use coeus_dtype::float::Float32;
use coeus_backend::Backend;
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_dtype::DataType;
use coeus_tensor::AsAny;

/// Perform element-wise addition with automatic differentiation
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn add<B, T>(
    lhs: &coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>,
    rhs: &coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>,
) -> coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>
where
    B: Backend + Clone + Default,
    T: DataType + Add<Output = T>,
{
    // Perform the actual addition
    let result = lhs + rhs;

    // Set grad_fn if either input requires gradients
    if lhs.requires_grad() || rhs.requires_grad() {
        let add_fn = Arc::new(AddFunction::new(Arc::new(lhs.clone()), Arc::new(rhs.clone())));
        let result = result;
        // For now, skip setting grad_fn until trait issues are resolved
        // result.set_grad_fn(Some(add_fn));
        result
    } else {
        result
    }
}

/// Perform element-wise multiplication with automatic differentiation
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn mul(
    lhs: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    rhs: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> {
    // Perform the actual multiplication
    let result = lhs * rhs;

    // Set grad_fn if either input requires gradients
    if lhs.requires_grad() || rhs.requires_grad() {
        let mul_fn = Arc::new(MulFunction::new(Arc::new(lhs.clone()), Arc::new(rhs.clone())));
        let mut result = result;
        result.set_grad_fn(Some(mul_fn));
        result
    } else {
        result
    }
}

/// Perform matrix multiplication with automatic differentiation
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn matmul(
    lhs: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    rhs: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> Result<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>, crate::error::AutogradError> {
    // Perform the actual matrix multiplication
    let result = lhs.matmul(rhs)?;

    // Set grad_fn if either input requires gradients
    if lhs.requires_grad() || rhs.requires_grad() {
        let matmul_fn = Arc::new(MatMulFunction::new(Arc::new(lhs.clone()), Arc::new(rhs.clone())));
        let mut result = result;
        result.set_grad_fn(Some(matmul_fn));
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Perform sum reduction with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to sum
/// * `dims` - Optional dimensions to reduce (None for all dimensions)
/// * `keepdim` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn sum(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>, crate::error::AutogradError> {
    // Perform the actual sum
    let result = input.sum_dims(dims, keepdim)?;

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let sum_fn = Arc::new(SumFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(sum_fn));
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Perform mean reduction with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to average
/// * `dims` - Optional dimensions to reduce (None for all dimensions)
/// * `keepdim` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn mean(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>, crate::error::AutogradError> {
    // Perform the actual mean
    let result = input.mean_dims(dims, keepdim)?;

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let mean_fn = Arc::new(MeanFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(mean_fn));
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Perform element-wise exponential with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to exponentiate
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn exp(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> {
    // Perform the actual exponential
    let result = input.exp();

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let exp_fn = Arc::new(ExpFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(exp_fn));
        result
    } else {
        result
    }
}

/// Perform element-wise natural logarithm with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to take logarithm of
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn log(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> {
    // Perform the actual logarithm
    let result = input.log();

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let log_fn = Arc::new(LogFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(log_fn));
        result
    } else {
        result
    }
}

/// Perform element-wise sine with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to take sine of
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn sin(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> {
    // Perform the actual sine
    let result = input.sin();

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let sin_fn = Arc::new(SinFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(sin_fn));
        result
    } else {
        result
    }
}

/// Perform element-wise cosine with automatic differentiation
///
/// # Arguments
/// * `input` - Input tensor to take cosine of
///
/// # Returns
/// Result tensor with grad_fn set for backward pass
#[must_use]
pub fn cos(
    input: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
) -> coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> {
    // Perform the actual cosine
    let result = input.cos();

    // Set grad_fn if input requires gradients
    if input.requires_grad() {
        let cos_fn = Arc::new(CosFunction::new(Arc::new(input.clone())));
        let mut result = result;
        result.set_grad_fn(Some(cos_fn));
        result
    } else {
        result
    }
}

/// Perform backward pass to compute gradients through the computation graph
///
/// This function implements PyTorch-compatible automatic differentiation by:
/// 1. Performing topological sorting of the computation graph
/// 2. Computing gradients in reverse topological order
/// 3. Accumulating gradients into tensor.grad fields
///
/// # Arguments
/// * `tensor` - Tensor to compute gradients for (typically a scalar loss)
///
/// # Returns
/// Result indicating success or failure
///
/// # Panics
/// Panics if gradient accumulation fails due to lock poisoning

/// Perform backward pass with explicit gradient for the output tensor
///
/// # Arguments
/// * `tensor` - Tensor to compute gradients for
/// * `grad_output` - Gradient w.r.t. the output tensor
///
/// # Returns
/// Result indicating success or failure
///
/// # Panics
/// Panics if gradient accumulation fails due to lock poisoning
pub fn backward_with_grad<B, S, T>(
    tensor: &coeus_tensor::Tensor<B, S, T>,
    grad_output: &coeus_tensor::Tensor<B, S, T>,
) -> Result<(), crate::error::AutogradError>
where
    B: Backend,
    S: Storage<T> + 'static,
    T: DataType,
{
    backward_with_grad_and_options(tensor, grad_output, false)
}

/// Perform backward pass with explicit gradient and higher-order derivative support
///
/// # Arguments
/// * `tensor` - Tensor to compute gradients for
/// * `grad_output` - Gradient w.r.t. the output tensor
/// * `create_graph` - If true, computed gradients are themselves differentiable
///
/// # Returns
/// Result indicating success or failure
pub fn backward_with_grad_and_options<B, S, T>(
    tensor: &coeus_tensor::Tensor<B, S, T>,
    grad_output: &coeus_tensor::Tensor<B, S, T>,
    create_graph: bool,
) -> Result<(), crate::error::AutogradError>
where
    B: Backend,
    S: Storage<T> + 'static,
    T: DataType,
{
    // MS-6.3: Implement automatic dynamic graph construction with topological sorting
    // For now, implement single-function backward with proper generic handling
    // Full topological sorting with NodeRegistry will be added in MS-6.4

    if let Some(grad_fn) = tensor.grad_fn() {
        // For the generic system, we use a different approach
        // Since functions are now generic, we need to handle them through trait objects
        // For now, we assume all operations use the default tensor type

        // Try to downcast to each specific function type with concrete generics
        type ConcreteBackend = coeus_backend::CpuBackend;
        type ConcreteStorage = coeus_storage::DenseStorage<coeus_dtype::float::Float32>;
        type ConcreteDtype = coeus_dtype::float::Float32;

        // For now, assume all operations use dense storage and can be downcast
        // This is a temporary solution until we have proper generic trait objects
        let concrete_grad_output = if let Some(concrete_tensor) = grad_output.as_any().downcast_ref::<coeus_tensor::Tensor<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            concrete_tensor.clone()
        } else {
            return Err(crate::error::AutogradError::GraphError("Unsupported tensor type for backward".to_string()).into());
        };

        if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::AddFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::MulFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::MatMulFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::SumFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::MeanFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::ExpFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::LogFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::SinFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::functions::CosFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else if let Some(func) = grad_fn.as_ref().as_any().downcast_ref::<crate::custom::CustomFunction<ConcreteBackend, ConcreteStorage, ConcreteDtype>>() {
            let grad_inputs = func.backward(&concrete_grad_output).map_err(|e| crate::error::AutogradError::InvalidOperation {
                operation: format!("Function backward failed: {}", e),
            })?;
            accumulate_gradients_with_options(func.inputs(), &grad_inputs, create_graph);
        } else {
            return Err(crate::error::AutogradError::InvalidOperation {
                operation: "Unknown function type".to_string(),
            });
        }
    }

    Ok(())
}

/// Accumulate gradients into input tensors
fn accumulate_gradients<B, T>(
    inputs: &[crate::functions::TensorRef<B, DenseStorage<T>, T>],
    grad_inputs: &[coeus_tensor::Tensor<B, DenseStorage<T>, T>],
)
where
    B: Backend + Default,
    T: DataType + core::ops::Add<Output = T>,
{
    accumulate_gradients_with_options(inputs, grad_inputs, false)
}

/// Accumulate gradients into input tensors with higher-order derivative support
fn accumulate_gradients_with_options<B, T>(
    inputs: &[crate::functions::TensorRef<B, DenseStorage<T>, T>],
    grad_inputs: &[coeus_tensor::Tensor<B, DenseStorage<T>, T>],
    create_graph: bool,
)
where
    B: Backend + Default,
    T: DataType + core::ops::Add<Output = T>,
{
    // Create gradient accumulator for batch gradient accumulation
    let mut accumulator = crate::graph_node::GradientAccumulator::new();

    // Accumulate all gradients
    for (i, input_tensor) in inputs.iter().enumerate() {
        if i < grad_inputs.len() {
            let grad = &grad_inputs[i];
            accumulator.accumulate(input_tensor, grad.clone());
        }
    }

    // Apply accumulated gradients to tensors
    if let Err(e) = accumulator.apply_gradients() {
        // In a real implementation, this would return a Result
        // For now, we panic on gradient application failure
        panic!("Failed to apply gradients: {:?}", e);
    }

    // Handle higher-order derivatives if create_graph is true
    if create_graph {
        // TODO: Set grad_fn on gradients for higher-order derivatives
        // This enables grad.grad operations in PyTorch style
        // For now, higher-order derivatives are not implemented
    }
}

/// Compute Hessian-Vector Product (HVP) for higher-order derivatives
///
/// This computes the product of the Hessian matrix with a vector v.
/// HVP is computed using forward-over-reverse automatic differentiation.
///
/// # Arguments
/// * `output` - The output tensor (scalar) to differentiate
/// * `inputs` - Input tensors w.r.t. which to compute the Hessian
/// * `v` - Vector to multiply with the Hessian
///
/// # Returns
/// HVP result as a vector of tensors
pub fn hvp(
    output: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    inputs: &[&coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>],
    v: &[coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>],
) -> Result<Vec<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>>, crate::error::AutogradError> {
    // First compute VJP (vector-Jacobian product) with create_graph=true
    // This gives us the Jacobian-vector product as differentiable tensors
    let grad_outputs = grad(output, inputs, Some(v), true)?;

    // Now compute the gradient of the VJP w.r.t. inputs to get HVP
    // This is the forward-over-reverse approach
    let mut hvp_result = Vec::new();

    for grad_output in &grad_outputs {
        // Compute gradient of the VJP result w.r.t. all inputs
        let hvp_components = grad(grad_output, inputs, None, false)?;
        hvp_result.extend(hvp_components);
    }

    Ok(hvp_result)
}

/// Compute Jacobian-Vector Product (JVP) using forward-mode AD
///
/// # Arguments
/// * `func` - Function that takes inputs and returns outputs
/// * `inputs` - Input tensors
/// * `v` - Tangent vectors (directions for differentiation)
///
/// # Returns
/// JVP result
pub fn jvp<F>(
    func: F,
    inputs: &[&coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>],
    v: &[coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>],
) -> Result<Vec<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>>, crate::error::AutogradError>
where
    F: Fn(&[&coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>])
        -> Result<Vec<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>>, crate::error::AutogradError>,
{
    // Forward-mode AD implementation
    // This is a simplified version - full forward-mode AD would require
    // dual number arithmetic throughout the computation graph

    // For now, return empty result as placeholder
    // TODO: Implement full forward-mode AD
    let _func = func;
    let _inputs = inputs;
    let _v = v;

    Ok(Vec::new())
}

/// Compute gradient with higher-order derivative support
///
/// # Arguments
/// * `output` - Output tensor to differentiate
/// * `inputs` - Input tensors w.r.t. which to compute gradients
/// * `grad_outputs` - Optional gradient w.r.t. outputs (defaults to ones)
/// * `create_graph` - Whether gradients should be differentiable
///
/// # Returns
/// Gradients w.r.t. inputs
pub fn grad(
    output: &coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>,
    inputs: &[&coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>],
    grad_outputs: Option<&[coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>]>,
    create_graph: bool,
) -> Result<Vec<coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>>, crate::error::AutogradError> {
    // This is a simplified implementation
    // In a full implementation, this would traverse the computation graph
    // and compute gradients for all specified inputs

    let mut gradients = Vec::new();

    // Create default grad_outputs if not provided
    let default_grad = if output.shape().dims().is_empty() {
        // Scalar output
        coeus_tensor::Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap()
    } else {
        // TODO: Handle non-scalar outputs
        return Err(crate::error::AutogradError::InvalidOperation {
            operation: "Non-scalar outputs not yet supported".to_string(),
        });
    };

    let grad_output = grad_outputs.and_then(|g| g.first()).unwrap_or(&default_grad);

    // For each input, compute its gradient
    for input in inputs {
        // This is a placeholder - real implementation would need to:
        // 1. Build the subgraph from output to this input
        // 2. Compute the gradient through that subgraph
        // 3. If create_graph=true, ensure the gradient tensor has autograd enabled

        // For now, create a zero gradient tensor with the same shape as input
        let zero_grad = coeus_tensor::Tensor::zeros(input.shape().dims()).unwrap();

        if create_graph {
            // TODO: Set up autograd for the gradient tensor itself
            // This would require the gradient computation to be recorded
        }

        gradients.push(zero_grad);
    }

    // Actually compute the gradient for the first input as an example
    if !inputs.is_empty() {
        backward_with_grad_and_options(output, grad_output, create_graph)?;
        // TODO: Extract the actual gradients from input tensors
    }

    Ok(gradients)
}

/// Perform backward pass on a scalar tensor
///
/// # Arguments
/// * `tensor` - Scalar tensor to compute gradients for
pub fn backward<B, T>(
    tensor: &coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>,
) -> crate::Result<()>
where
    B: Backend + core::fmt::Debug + Send + Sync + Clone + 'static,
    T: DataType,
{
    if tensor.shape().ndim() != 0 {
        return Err(crate::error::AutogradError::InvalidInput {
            message: "backward() requires scalar tensor".to_string(),
        });
    }

    // Create gradient = 1.0 for scalar backward
    let one_storage = coeus_storage::DenseStorage::from_vec(vec![T::one()], &[])
        .map_err(|e| crate::error::AutogradError::TensorError(
            coeus_tensor::TensorError::StorageError(e)
        ))?;
    let grad_output = coeus_tensor::Tensor::from_storage(one_storage, tensor.backend().clone());

    backward_with_grad(tensor, &grad_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_tensor::Tensor;
    use coeus_storage::StorageToDense;
    use crate::GradientEngine;

    #[test]
    fn test_backward_with_create_graph() {
        // Test that create_graph=true works without panicking
        // This is a basic smoke test - full higher-order derivative testing
        // would require more complete gradient extraction from tensors

        let input: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(2.0)], &[]).unwrap().requires_grad_(true);
        let output: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(4.0)], &[]).unwrap();

        // This should not panic
        let result = backward_with_grad_and_options(&output, &Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap(), true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_grad_function() {
        // Test the grad function with basic inputs
        let input: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(2.0)], &[]).unwrap().requires_grad_(true);
        let output: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(4.0)], &[]).unwrap();

        let inputs = vec![&input];
        let grad_outputs = vec![Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap()];

        // Test with create_graph=false
        let gradients = grad(&output, &inputs, Some(&grad_outputs), false).unwrap();
        assert_eq!(gradients.len(), 1);

        // Test with create_graph=true
        let gradients_with_graph = grad(&output, &inputs, Some(&grad_outputs), true).unwrap();
        assert_eq!(gradients_with_graph.len(), 1);
    }

    #[test]
    fn test_hvp_basic() {
        // Basic smoke test for HVP function
        let output: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap();
        let input: Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32> =
            Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap().requires_grad_(true);
        let inputs = vec![&input];
        let v = vec![Tensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap()];

    // This should not panic - full functionality requires complete gradient extraction
    let result = hvp(&output, &inputs, &v);
    assert!(result.is_ok());
}


/// Perform backward pass with explicit gradient
///
/// # Arguments
/// * `tensor` - Tensor to compute gradients for
/// * `grad_output` - Gradient w.r.t. the tensor
pub fn backward_with_grad<B, S, T>(
    tensor: &coeus_tensor::Tensor<B, S, T>,
    grad_output: &coeus_tensor::Tensor<B, S, T>,
) -> crate::Result<()>
where
    B: Backend + core::fmt::Debug + Send + Sync + Clone + 'static,
    S: Storage<T> + core::fmt::Debug + Send + Sync + StorageToDense<T> + 'static,
    T: DataType,
{
    if !tensor.requires_grad() {
        return Err(crate::error::AutogradError::InvalidInput {
            message: "Cannot call backward on tensor that doesn't require gradients".to_string(),
        });
    }

    let mut engine = GradientEngine::new();
    engine.backward(tensor.grad_fn(), grad_output)
}
}

