use crate::computation_graph::GradientEngine;
use crate::error::{AutogradError, Result};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

// Re-export tensor operations
pub use crate::loss::nll_loss;
pub use crate::tensor_ops::{
    add, cos, div, exp, log, matmul, mean, mul, neg, pow, reshape, sin, sub, sum, transpose,
};

/// Compute gradients for the given tensor.
///
/// This function triggers the backward pass of the automatic differentiation engine.
/// It computes the gradient of the tensor with respect to graph leaves.
///
/// # Arguments
/// * `tensor` - The tensor to compute gradients for (usually a scalar loss)
/// * `grad_tensor` - Optional gradient w.r.t. the tensor (defaults to 1.0)
/// * `retain_graph` - Whether to retain the computation graph for multiple backward passes
/// * `create_graph` - Whether to create a graph for higher-order derivatives
#[allow(clippy::missing_errors_doc)]
pub fn backward<B, S, T>(
    tensor: &Tensor<B, S, T>,
    grad_tensor: Option<Tensor<B, S, T>>,
    _retain_graph: bool,
    _create_graph: bool,
) -> Result<()>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType + num_traits::One,
{
    let grad = if let Some(g) = grad_tensor {
        g
    } else {
        let dims = tensor.shape().dims();
        let size = dims.iter().product();
        let data = vec![T::one(); size];
        Tensor::from_vec_with_backend(data, dims, tensor.backend().clone()).map_err(|e| {
            AutogradError::InvalidInput {
                message: e.to_string(),
            }
        })?
    };

    let mut engine = GradientEngine::new();
    // TODO: Handle retain_graph and create_graph in GradientEngine
    engine.backward(tensor.grad_fn(), &grad)
}

/// Compute gradients for the given tensor with a specific gradient.
#[allow(clippy::missing_errors_doc)]
pub fn backward_with_grad<B, S, T>(tensor: &Tensor<B, S, T>, grad: Tensor<B, S, T>) -> Result<()>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType + num_traits::One,
{
    backward(tensor, Some(grad), false, false)
}

/// Compute gradients for the given tensor with options.
#[allow(clippy::missing_errors_doc)]
pub fn backward_with_grad_and_options<B, S, T>(
    tensor: &Tensor<B, S, T>,
    grad: Tensor<B, S, T>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<()>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType + num_traits::One,
{
    backward(tensor, Some(grad), retain_graph, create_graph)
}

/// Compute and return the sum of gradients of outputs w.r.t. the inputs.
///
/// # Arguments
/// * `outputs` - The output tensors
/// * `inputs` - The input tensors to compute gradients for
/// * `grad_outputs` - The gradient w.r.t. each output (defaults to 1.0)
/// * `retain_graph` - Whether to retain the graph
/// * `create_graph` - Whether to create a graph for higher order derivatives
/// * `allow_unused` - Whether to allow inputs that are not part of the graph (returns zeros)
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
pub fn grad<B, S, T>(
    outputs: &[Tensor<B, S, T>],
    inputs: &[Tensor<B, S, T>],
    grad_outputs: Option<Vec<Tensor<B, S, T>>>,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool,
) -> Result<Vec<Tensor<B, S, T>>>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType + num_traits::One + num_traits::Zero + Copy,
{
    if let Some(grads) = &grad_outputs {
        if grads.len() != outputs.len() {
            return Err(AutogradError::InvalidInput {
                message: format!(
                    "grad_outputs length {} does not match outputs length {}",
                    grads.len(),
                    outputs.len()
                ),
            });
        }
    }

    // Run backward for each output
    for (i, output) in outputs.iter().enumerate() {
        let grad_out = grad_outputs.as_ref().map(|grads| grads[i].clone());

        // If we have multiple outputs, we must retain graph for intermediate backward passes
        let retain = if i < outputs.len() - 1 {
            true
        } else {
            retain_graph
        };

        backward(output, grad_out, retain, create_graph)?;
    }

    // Collect gradients
    let mut results = Vec::new();
    for input in inputs {
        match input.grad() {
            Ok(dense_grad) => {
                // Convert DenseStorage to S
                // dense_grad is Tensor<B, DenseStorage<T>, T>
                // We use as_slice() to get data and recreate tensor with backend B and storage S
                let data = dense_grad.as_slice().to_vec();
                let dims = dense_grad.shape().dims();
                let grad_s = Tensor::from_vec_with_backend(data, dims, input.backend().clone())
                    .map_err(|e| AutogradError::InvalidInput {
                        message: e.to_string(),
                    })?;
                results.push(grad_s);
            }
            Err(_) => {
                if allow_unused {
                    let size = input.shape().size();
                    let data = vec![T::zero(); size];
                    let zeros_s = Tensor::from_vec_with_backend(
                        data,
                        input.shape().dims(),
                        input.backend().clone(),
                    )
                    .map_err(|e| AutogradError::InvalidInput {
                        message: e.to_string(),
                    })?;
                    results.push(zeros_s);
                } else {
                    return Err(AutogradError::GraphError(
                        "Input gradient not available".to_string(),
                    ));
                }
            }
        }
    }

    Ok(results)
}

/// Compute Hessian-vector product
///
/// Computes the product of the Hessian matrix of `func` at `inputs` with vector `v`.
/// This is done efficiently using two backward passes (or one forward and one backward).
///
/// # Arguments
/// * `_func` - Function to compute Hessian for
/// * `_inputs` - Input tensors
/// * `_v` - Vector to multiply Hessian with
#[allow(clippy::missing_errors_doc)]
pub fn hvp<B, S, T>(
    _func: impl Fn(&[Tensor<B, S, T>]) -> Result<Tensor<B, S, T>>,
    _inputs: &[Tensor<B, S, T>],
    _v: &[Tensor<B, S, T>],
) -> Result<Vec<Tensor<B, S, T>>>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType,
{
    // TODO: Implement Hessian-vector product
    // 1. Compute gradients of func w.r.t inputs (g = grad(func(inputs)))
    // 2. Compute gradients of (g * v) w.r.t inputs
    Err(AutogradError::NotImplemented {
        operation: "hvp".to_string(),
    })
}

/// Compute Jacobian-vector product
///
/// Computes the product of the Jacobian matrix of `func` at `inputs` with vector `v`.
/// This is done efficiently using forward mode AD (or dual numbers), but here we
/// might simulate it or use double backward.
///
/// # Arguments
/// * `_func` - Function to compute Jacobian for
/// * `_inputs` - Input tensors
/// * `_v` - Vector to multiply Jacobian with
#[allow(clippy::missing_errors_doc)]
pub fn jvp<B, S, T>(
    _func: impl Fn(&[Tensor<B, S, T>]) -> Result<Tensor<B, S, T>>,
    _inputs: &[Tensor<B, S, T>],
    _v: &[Tensor<B, S, T>],
) -> Result<Vec<Tensor<B, S, T>>>
where
    B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
    S: Storage<T>
        + core::fmt::Debug
        + Send
        + Sync
        + 'static
        + StorageToDense<T>
        + StorageFromVec<T>,
    T: DataType,
{
    // TODO: Implement Jacobian-vector product
    // Ideally use forward mode AD
    Err(AutogradError::NotImplemented {
        operation: "jvp".to_string(),
    })
}
