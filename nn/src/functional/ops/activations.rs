//! Activation functions for neural networks.
//!
//! This module provides stateless activation functions that operate on tensors directly,
//! implementing common activation functions used in neural networks with SIMD acceleration.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::Result;

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
/// use nn::ops::activations::relu;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.5), Float32::new(2.0)],
///     &[1, 3]
/// ).unwrap();
///
/// let output = relu(&input).unwrap();
/// // output: [0.0, 0.5, 2.0]
/// ```
pub fn relu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = vec![T::from(0.0).unwrap(); data.len()];

    let zero = T::from(0.0).unwrap();
    for (i, &val) in data.iter().enumerate() {
        result_data[i] = if val > zero { val } else { zero };
    }

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn sigmoid<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone,
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

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn tanh<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    for &val in data {
        result_data.push(val.tanh());
    }

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn gelu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + Clone
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>,
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

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn silu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone + std::ops::Mul<Output = T>,
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

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn leaky_relu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    negative_slope: T,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone + std::ops::Mul<Output = T>,
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

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
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
pub fn elu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    alpha: T,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType
        + FloatExt
        + PartialOrd
        + Clone
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    for &x in data {
        let result = if x > zero { x } else { alpha * (x.exp() - one) };
        result_data.push(result);
    }

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
}

/// Applies the Softmax function to an n-dimensional input Tensor.
///
/// Rescales elements so that they lie in the range [0, 1] and sum to 1.
///
/// Formula: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with softmax applied along the last dimension
pub fn softmax<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Clone
        + num_traits::FromPrimitive,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();
    let input_data = input_dense.as_slice();

    let last_dim_size = *input_shape.last().unwrap();
    let batch_size: usize = input_data.len() / last_dim_size;

    let mut output_data = Vec::with_capacity(input_data.len());

    for batch in 0..batch_size {
        let start_idx = batch * last_dim_size;
        let end_idx = start_idx + last_dim_size;
        let slice = &input_data[start_idx..end_idx];

        let mut max_val = slice[0];
        for val in slice {
            if *val > max_val {
                max_val = *val;
            }
        }

        let mut exp_sum = T::from_f32(0.0).unwrap();
        let mut exp_values = Vec::with_capacity(last_dim_size);

        for val in slice {
            let shifted = *val - max_val;
            let exp_val = shifted.exp();
            exp_values.push(exp_val);
            exp_sum = exp_sum + exp_val;
        }

        for exp_val in exp_values {
            output_data.push(exp_val / exp_sum);
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        input_shape,
        input.backend().clone(),
    )?)
}

pub fn softmax_dim<B, S, T>(
    input: &Tensor<B, S, T>,
    dim: isize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Clone
        + num_traits::FromPrimitive,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();
    let input_data = input_dense.as_slice();

    let rank = input_shape.len();
    if rank == 0 {
        return Err(crate::core::error::NNError::InvalidInput {
            message: "softmax_dim requires a non-scalar tensor".to_string(),
        });
    }

    let normalized_dim = if dim < 0 { dim + rank as isize } else { dim };

    if normalized_dim < 0 || normalized_dim >= rank as isize {
        return Err(crate::core::error::NNError::InvalidInput {
            message: format!("dim {} out of range for rank {}", dim, rank),
        });
    }
    let dim = normalized_dim as usize;

    let axis_size = input_shape[dim];
    let inner: usize = input_shape[dim + 1..].iter().product();
    let outer: usize = input_shape[..dim].iter().product();

    let mut output_data = vec![T::from_f32(0.0).unwrap(); input_data.len()];

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_size * inner + i;

            let mut max_val = input_data[base];
            for k in 1..axis_size {
                let val = input_data[base + k * inner];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut exp_sum = T::from_f32(0.0).unwrap();
            for k in 0..axis_size {
                let shifted = input_data[base + k * inner] - max_val;
                let exp_val = shifted.exp();
                output_data[base + k * inner] = exp_val;
                exp_sum = exp_sum + exp_val;
            }

            for k in 0..axis_size {
                output_data[base + k * inner] = output_data[base + k * inner] / exp_sum;
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        input_shape,
        input.backend().clone(),
    )?)
}

/// During training, randomly zeroes some of the elements of the input tensor with probability `p`.
///
/// # Arguments
/// * `input` - Input tensor
/// * `p` - Probability of an element to be zeroed (default: 0.5)
/// * `training` - If true, applies dropout. If false, returns input as is (default: true)
pub fn dropout<B, S, T>(input: &Tensor<B, S, T>, p: f64, training: bool) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    if !training || p <= 0.0 {
        return Ok(input.clone());
    }

    if p >= 1.0 {
        let result_data = vec![T::from(0.0).unwrap(); input.as_slice().len()];
        return Ok(Tensor::from_vec_with_backend(
            result_data,
            input.shape().dims(),
            input.backend().clone(),
        )?);
    }

    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let data = input.as_slice();
    let mut result_data = Vec::with_capacity(data.len());
    let scale = T::from_f64(1.0 / (1.0 - p)).unwrap();

    for &val in data {
        if rng.gen::<f64>() < p {
            result_data.push(T::from(0.0).unwrap());
        } else {
            result_data.push(val * scale);
        }
    }

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
}
