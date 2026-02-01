//! Activation operations
//!
//! Stateless implementations of activation functions.
//! Most delegate to `tensor::ops`, others are implemented here.
//! This module acts as the Single Source of Truth for activation logic in the NN crate.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;
use crate::core::error::Result;

// Delegate to tensor::ops where possible

/// Applies the Rectified Linear Unit (ReLU) activation function.
pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static, // Added bounds
    T: DataType + FloatExt + PartialOrd + Clone + num_traits::Zero,
{
    // tensor::ops::relu usually preserves storage, but let's check or handle it safe
    // If it returns S, good. If not, we convert.
    // Assuming tensor::ops::relu returns generic S if supported, or Dense.
    // Error log didn't complain about relu, only sigmoid/tanh/gelu.
    Ok(tensor::ops::relu(input)?)
}

/// Applies the Sigmoid activation function.
pub fn sigmoid<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + Clone + std::ops::Neg<Output = T>,
{
    Ok(tensor::ops::sigmoid(input)?)
}

/// Applies the Hyperbolic Tangent (tanh) activation function.
pub fn tanh<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + Clone,
{
    Ok(tensor::ops::tanh(input)?)
}

/// Applies the Gaussian Error Linear Unit (GELU) activation function.
pub fn gelu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + Clone + num_traits::FromPrimitive, // Added FromPrimitive
{
    Ok(tensor::ops::gelu(input)?)
}

/// Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
pub fn leaky_relu<B, S, T>(input: &Tensor<B, S, T>, negative_slope: T) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + PartialOrd + Clone + num_traits::FromPrimitive,
{
    // tensor::ops::leaky_relu expects f64 slope
    let slope_f64 = negative_slope.to_f64().ok_or_else(|| crate::core::error::NNError::InvalidInput {
         message: "Failed to convert negative_slope to f64".to_string() 
    })?;
    Ok(tensor::ops::leaky_relu(input, slope_f64)?)
}

/// Applies the Softmax function.
/// Delegates to tensor::ops::softmax with dim=-1 (last dimension).
pub fn softmax<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + 'static + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + Clone + PartialOrd + 'static,
{
    Ok(tensor::ops::softmax(input, -1)?)
}

/// Applies Softmax along a specific dimension.
pub fn softmax_dim<B, S, T>(
    input: &Tensor<B, S, T>,
    dim: isize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + 'static + StorageFromVec<T> + StorageToDense<T> + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + Clone + PartialOrd + 'static,
{
    Ok(tensor::ops::softmax(input, dim as i64)?)
}

/// Applies Log-Softmax.
/// 
/// L(x) = log(softmax(x))
pub fn log_softmax<B, S, T>(
    input: &Tensor<B, S, T>,
    dim: isize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + 'static + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + Clone + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    let probs = tensor::ops::softmax(input, dim as i64)?;
    Ok(tensor::ops::log(&probs)?)
}


/// Applies Dropout.
pub fn dropout<B, S, T>(input: &Tensor<B, S, T>, p: f64, training: bool) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static + StorageToDense<T> + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + Clone + num_traits::FromPrimitive,
{
    if !training || p <= 0.0 {
        return Ok(input.clone());
    }
    if p >= 1.0 {
        let zero = T::from_f64(0.0).unwrap();
        let len = input.shape().size();
        let data = vec![zero; len];
        return Ok(Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend().clone())?);
    }

    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    
    // Fix NNError::BackendError
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());
    let scale = T::from_f64(1.0 / (1.0 - p)).unwrap();
    let zero = T::from_f64(0.0).unwrap();

    for &val in data {
        if rng.gen::<f64>() < p {
            result_data.push(zero);
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

/// Applies ELU activation.
pub fn elu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    alpha: T,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + PartialOrd + Clone,
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

/// Applies SiLU (Swish) activation.
pub fn silu<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());
    let one = T::from(1.0).unwrap();

    for &x in data {
        let sigmoid_x = one / (one + (-x).exp());
        result_data.push(x * sigmoid_x);
    }

    Ok(Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?)
}
