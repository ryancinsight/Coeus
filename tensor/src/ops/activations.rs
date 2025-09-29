//! Activation functions for tensors
//!
//! This module provides mathematical activation functions with automatic differentiation
//! support for the unified tensor architecture.

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use coeus_backend::Backend;

/// Compute the hyperbolic tangent of each element
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// New tensor with tanh values
///
/// # Example
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let tensor = Tensor::scalar(0.0);
/// let tanh_tensor = tensor.tanh().unwrap();
/// assert_eq!(tanh_tensor.item().unwrap(), 0.0);
/// ```
pub fn tanh<T: FloatDtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor
        .data()
        .iter()
        .map(|x| x.tanh())
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    // Propagate requires_grad flag and setup autograd graph
    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂tanh(x)/∂x = 1 - tanh(x)^2)
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| crate::Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let tanh_node = context.create_node_with_data(Operation::Tanh, vec![input_node], vec![input_data_f64]);
            result.node = Some(tanh_node);
        });
    }

    Ok(result)
}

/// Compute the sigmoid of each element
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// New tensor with sigmoid values
///
/// # Example
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let tensor = Tensor::scalar(0.0);
/// let sigmoid_tensor = tensor.sigmoid().unwrap();
/// assert_eq!(sigmoid_tensor.item().unwrap(), 0.5);
/// ```
pub fn sigmoid<T: FloatDtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor
        .data()
        .iter()
        .map(|x| T::one() / (T::one() + (-*x).exp()))
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    // Propagate requires_grad flag and setup autograd graph
    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x)))
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| crate::Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let sigmoid_node = context.create_node_with_data(Operation::Sigmoid, vec![input_node], vec![input_data_f64]);
            result.node = Some(sigmoid_node);
        });
    }

    Ok(result)
}

/// Compute the ReLU (Rectified Linear Unit) of each element
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// New tensor with ReLU values
pub fn relu<T: FloatDtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor
        .data()
        .iter()
        .map(|x| if *x > T::zero() { *x } else { T::zero() })
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂relu(x)/∂x = 1 if x > 0 else 0)
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| crate::Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let relu_node = context.create_node_with_data(Operation::Relu, vec![input_node], vec![input_data_f64]);
            result.node = Some(relu_node);
        });
    }

    Ok(result)
}

/// Compute the Leaky ReLU of each element
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `negative_slope` - Slope for negative values (default: 0.01)
///
/// # Returns
/// New tensor with Leaky ReLU values
pub fn leaky_relu<T: FloatDtype, B: Backend<T> + Clone>(
    tensor: &Tensor<T, B>,
    negative_slope: T
) -> Result<Tensor<T, B>> {
    let data = tensor
        .data()
        .iter()
        .map(|x| if *x > T::zero() { *x } else { *x * negative_slope })
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Compute the ELU (Exponential Linear Unit) of each element
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `alpha` - Alpha parameter (default: 1.0)
///
/// # Returns
/// New tensor with ELU values
pub fn elu<T: FloatDtype, B: Backend<T> + Clone>(
    tensor: &Tensor<T, B>,
    alpha: T
) -> Result<Tensor<T, B>> {
    let data = tensor
        .data()
        .iter()
        .map(|x| if *x > T::zero() { *x } else { alpha * (x.exp() - T::one()) })
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Compute the GELU (Gaussian Error Linear Unit) of each element
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// New tensor with GELU values
pub fn gelu<T: FloatDtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    let data = tensor
        .data()
        .iter()
        .map(|x| {
            let x_f64 = Dtype::to_f64(x).unwrap_or(0.0);
            let result = 0.5 * x_f64 * (1.0 + (x_f64 / (2.0_f64).sqrt()).tanh());
            <T as Dtype>::from_f64(result).unwrap_or(T::zero())
        })
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Compute the softmax of each element along specified dimension
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Dimension along which to compute softmax
///
/// # Returns
/// New tensor with softmax values
pub fn softmax<T: FloatDtype, B: Backend<T> + Clone>(
    tensor: &Tensor<T, B>,
    dim: usize
) -> Result<Tensor<T, B>> {
    if dim >= tensor.ndim() {
        return Err(TensorError::InvalidDimension {
            dim,
            max_dim: tensor.ndim() - 1,
        });
    }

    let mut result_data = Vec::with_capacity(tensor.numel());

    // Get shape information
    let shape = tensor.shape();
    let stride = shape[dim];
    let outer_stride = shape[..dim].iter().product::<usize>();
    let inner_stride = shape[dim + 1..].iter().product::<usize>();

    for outer_idx in 0..outer_stride {
        for inner_idx in 0..inner_stride {
            // Find max for numerical stability
            let mut max_val = T::neg_infinity();
            for i in 0..stride {
                let idx = outer_idx * stride * inner_stride + i * inner_stride + inner_idx;
                let val = tensor.data()[idx];
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp and sum
            let mut exp_sum = T::zero();
            let mut exp_values = Vec::with_capacity(stride);

            for i in 0..stride {
                let idx = outer_idx * stride * inner_stride + i * inner_stride + inner_idx;
                let val = tensor.data()[idx];
                let exp_val = (val - max_val).exp();
                exp_values.push(exp_val);
                exp_sum = exp_sum + exp_val;
            }

            // Compute softmax values
            for exp_val in exp_values {
                result_data.push(exp_val / exp_sum);
            }
        }
    }

    let mut result = Tensor::from_vec(tensor.backend.clone(), result_data, shape.to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Compute the log-softmax of each element along specified dimension
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Dimension along which to compute log-softmax
///
/// # Returns
/// New tensor with log-softmax values
pub fn log_softmax<T: FloatDtype, B: Backend<T> + Clone>(
    tensor: &Tensor<T, B>,
    dim: usize
) -> Result<Tensor<T, B>> {
    let softmax_tensor = softmax(tensor, dim)?;
    let data = softmax_tensor
        .data()
        .iter()
        .map(|x| x.ln())
        .collect();

    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;

    #[test]
    fn test_tanh() {
        let backend = CpuBackend::new();
        let tensor = Tensor::from_vec(backend, vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = tanh(&tensor).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data()[0], 0.0); // tanh(0) = 0
        assert!((result.data()[1] as f64 - 0.7616).abs() < 1e-4); // tanh(1) ≈ 0.7616
        assert!((result.data()[2] as f64 + 0.7616).abs() < 1e-4); // tanh(-1) ≈ -0.7616
    }

    #[test]
    fn test_sigmoid() {
        let backend = CpuBackend::new();
        let tensor = Tensor::from_vec(backend, vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = sigmoid(&tensor).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data()[0], 0.5); // sigmoid(0) = 0.5
        assert!((result.data()[1] as f64 - 0.7311).abs() < 1e-4); // sigmoid(1) ≈ 0.7311
        assert!((result.data()[2] as f64 - 0.2689).abs() < 1e-4); // sigmoid(-1) ≈ 0.2689
    }

    #[test]
    fn test_relu() {
        let backend = CpuBackend::new();
        let tensor = Tensor::from_vec(backend, vec![1.0, -1.0, 0.0], vec![3]).unwrap();
        let result = relu(&tensor).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 0.0);
        assert_eq!(result.data()[2], 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let backend = CpuBackend::new();
        let tensor = Tensor::from_vec(backend, vec![1.0, -1.0, 0.0], vec![3]).unwrap();
        let result = leaky_relu(&tensor, 0.01).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], -0.01);
        assert_eq!(result.data()[2], 0.0);
    }

    #[test]
    fn test_gelu() {
        let backend = CpuBackend::new();
        let tensor = Tensor::from_vec(backend, vec![1.0, -1.0, 0.0], vec![3]).unwrap();
        let result = gelu(&tensor).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert!((result.data()[0] as f64 - 0.8412).abs() < 1e-4); // GELU(1) ≈ 0.8412
        assert!((result.data()[1] as f64 + 0.1587).abs() < 1e-4); // GELU(-1) ≈ -0.1587
        assert_eq!(result.data()[2], 0.0); // GELU(0) = 0
    }
}
