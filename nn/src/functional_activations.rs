//! Activation functions for neural networks.
//!
//! This module provides stateless activation functions that operate on tensors directly,
//! implementing common activation functions used in neural networks with SIMD acceleration.

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::{Tensor, relu_simd};

use crate::error::{NNError, Result};

/// Applies the Rectified Linear Unit (ReLU) activation function.
///
/// Formula: `max(0, x)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with ReLU applied element-wise
///
/// # Examples
/// ```rust
/// use coeus_nn::functional_activations::relu;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.5), Float32::new(2.0)],
///     &[1, 3]
/// ).unwrap();
///
/// let output = relu(&input).unwrap();
/// // output: [0.0, 0.5, 2.0]
/// ```
pub fn relu<T: DataType + PartialOrd>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = vec![T::from(0.0).unwrap(); data.len()];

    // Use SIMD acceleration for f32 tensors
    if let Ok(cpu_tensor) = input_dense.as_any().downcast_ref::<Tensor<coeus_backend::CpuBackend, DenseStorage<f32>, f32>>() {
        relu_simd(cpu_tensor.as_slice(), &mut result_data);
    } else {
        // Fallback to scalar implementation for other types
        let zero = T::from(0.0).unwrap();
        for (i, &val) in data.iter().enumerate() {
            result_data[i] = if val > zero { val } else { zero };
        }
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Sigmoid activation function.
///
/// Formula: `1 / (1 + exp(-x))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with sigmoid applied element-wise
pub fn sigmoid<T: DataType + FloatExt + std::ops::Neg<Output = T>>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let one = T::from(1.0).unwrap();
    for &val in data {
        let neg_val = -val;
        let exp_neg = neg_val.exp();
        let denom = one + exp_neg;
        result_data.push(one / denom);
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Hyperbolic Tangent (tanh) activation function.
///
/// Formula: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with tanh applied element-wise
pub fn tanh<T: DataType + FloatExt>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    for &val in data {
        result_data.push(val.tanh());
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Gaussian Error Linear Unit (GELU) activation function.
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with GELU applied element-wise
pub fn gelu<T: DataType + FloatExt + std::ops::Neg<Output = T>>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let sqrt_2_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
    let coeff = T::from(0.044715).unwrap();

    for &x in data {
        let x_cubed = x * x * x;
        let inner = sqrt_2_pi * (x + coeff * x_cubed);
        let tanh_inner = inner.tanh();
        let gelu_val = half * x * (one + tanh_inner);
        result_data.push(gelu_val);
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Sigmoid Linear Unit (SiLU) activation function.
///
/// Formula: `x * sigmoid(x)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with SiLU applied element-wise
pub fn silu<T: DataType + FloatExt + std::ops::Neg<Output = T>>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let one = T::from(1.0).unwrap();
    for &x in data {
        let neg_x = -x;
        let exp_neg_x = neg_x.exp();
        let sigmoid_x = one / (one + exp_neg_x);
        let silu_val = x * sigmoid_x;
        result_data.push(silu_val);
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
///
/// Formula: `max(α * x, x)` where α is the negative slope
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `negative_slope` - Slope for negative inputs (default: 0.01)
///
/// # Returns
/// Tensor with Leaky ReLU applied element-wise
pub fn leaky_relu<T: DataType + FloatExt + PartialOrd>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
    negative_slope: T,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let zero = T::from(0.0).unwrap();
    for &val in data {
        let result = if val > zero {
            val
        } else {
            negative_slope * val
        };
        result_data.push(result);
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Applies the Exponential Linear Unit (ELU) activation function.
///
/// Formula: `x if x > 0 else α * (exp(x) - 1)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `alpha` - Scaling factor for negative inputs (default: 1.0)
///
/// # Returns
/// Tensor with ELU applied element-wise
pub fn elu<T: DataType + FloatExt + PartialOrd>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
    alpha: T,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    for &x in data {
        let result = if x > zero {
            x
        } else {
            alpha * (x.exp() - one)
        };
        result_data.push(result);
    }

    Tensor::from_vec(result_data, &input_dense.shape().dims())
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}
