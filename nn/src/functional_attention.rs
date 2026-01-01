//! Attention mechanisms for neural networks.
//!
//! This module provides stateless attention operations including
//! scaled dot-product attention for transformer architectures.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use std::ops::Neg;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::error::{NNError, Result};

/// Create a dropout mask for attention weights
///
/// # Arguments
/// * `tensor` - The tensor to create mask for
/// * `dropout_p` - Dropout probability (0.0 to 1.0)
///
/// # Returns
/// A binary mask tensor with the same shape as input
fn create_dropout_mask<B, S, T>(tensor: &Tensor<B, S, T>, dropout_p: f32) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + FromPrimitive,
{
    use rand::prelude::*;

    // Get tensor shape and size
    let shape = tensor.shape();
    let size = shape.dims().iter().product();

    // Create random mask using thread-local RNG for reproducibility
    let mut rng = rand::thread_rng();
    let mut mask_data = Vec::with_capacity(size);

    for _ in 0..size {
        // Keep element with probability (1 - dropout_p)
        let keep = rng.gen::<f32>() > dropout_p;
        mask_data.push(if keep {
            T::from_f32(1.0).unwrap()
        } else {
            T::from_f32(0.0).unwrap()
        });
    }

    // Create mask tensor with same shape
    Ok(Tensor::from_vec_with_backend(
        mask_data,
        shape.dims(),
        tensor.backend().clone(),
    )?)
}

/// Applies scaled dot-product attention mechanism.
///
/// This is the core attention operation used in transformer architectures.
/// It computes attention weights between queries and keys, then applies
/// these weights to the values.
///
/// Formula:
/// ```text
/// attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
/// ```
///
/// # Arguments
/// * `query` - Query tensor of shape `(..., seq_len_q, d_k)`
/// * `key` - Key tensor of shape `(..., seq_len_k, d_k)`
/// * `value` - Value tensor of shape `(..., seq_len_v, d_v)`
/// * `attn_mask` - Optional attention mask of shape `(..., seq_len_q, seq_len_k)`
/// * `dropout_p` - Dropout probability for attention weights (currently unused)
///
/// # Returns
/// Attention output tensor of shape `(..., seq_len_q, d_v)`
///
/// # Examples
/// ```rust
/// use nn::functional_attention::scaled_dot_product_attention;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let batch_size = 2;
/// let seq_len = 10;
/// let d_k = 64;
/// let d_v = 64;
///
/// let query = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_k]).unwrap();
/// let key = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_k]).unwrap();
/// let value = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_v]).unwrap();
///
/// let output = scaled_dot_product_attention(&query, &key, &value, None, 0.0, true).unwrap();
/// assert_eq!(output.shape().dims(), &[batch_size, seq_len, d_v]);
/// ```
pub fn scaled_dot_product_attention<B, S, T>(
    query: &Tensor<B, S, T>,
    key: &Tensor<B, S, T>,
    value: &Tensor<B, S, T>,
    attn_mask: Option<&Tensor<B, S, T>>,
    dropout_p: f64,
    training: bool,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + Neg<Output = T> + PartialOrd + FromPrimitive,
{
    let query_shape = query.shape().dims();
    let key_shape = key.shape().dims();
    let value_shape = value.shape().dims();

    if query_shape.len() < 2 || key_shape.len() < 2 || value_shape.len() < 2 {
        return Err(NNError::InvalidInput {
            message: "Query, key, and value must have at least 2 dimensions".to_string(),
        });
    }

    let _query_seq_len = query_shape[query_shape.len() - 2];
    let key_seq_len = key_shape[key_shape.len() - 2];
    let value_seq_len = value_shape[value_shape.len() - 2];
    let d_k = *query_shape.last().unwrap();

    if *key_shape.last().unwrap() != d_k {
        return Err(NNError::InvalidInput {
            message: format!(
                "Query and key last dimensions must match, got {} and {}",
                d_k,
                key_shape.last().unwrap()
            ),
        });
    }

    if key_seq_len != value_seq_len {
        return Err(NNError::InvalidInput {
            message: format!(
                "Key and value sequence lengths must match, got {} and {}",
                key_seq_len, value_seq_len
            ),
        });
    }

    // Validate attention mask if provided
    if let Some(mask) = attn_mask {
        let mask_shape = mask.shape().dims();
        let expected_mask_shape = [&query_shape[..query_shape.len() - 1], &[key_seq_len]].concat();
        if mask_shape != expected_mask_shape {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Attention mask shape {:?} does not match expected {:?}",
                    mask_shape, expected_mask_shape
                ),
            });
        }
    }

    // Convert to dense for computation
    let query_dense = query.to_dense_generic()?;
    let key_dense = key.to_dense_generic()?;
    let value_dense = value.to_dense_generic()?;

    // Compute attention scores: Q @ K^T
    let key_t = key_dense.transpose(key_shape.len() - 2, key_shape.len() - 1)?;
    let attn_scores = query_dense.matmul(&key_t)?;

    // Scale by sqrt(d_k)
    let scale_factor = T::from_f64((d_k as f64).sqrt()).unwrap();
    let attn_scores_scaled = attn_scores.mul_scalar(T::from_f64(1.0).unwrap() / scale_factor)?;

    // Apply attention mask if provided
    let attn_scores_masked = if let Some(mask) = attn_mask {
        let mask_dense = mask.to_dense_generic()?;
        // For simplicity, assume mask contains large negative values for masked positions
        // In a full implementation, this would handle different mask formats
        &attn_scores_scaled + &mask_dense
    } else {
        attn_scores_scaled
    };

    // Apply softmax to get attention weights
    let attn_weights = softmax(&attn_scores_masked)?;

    // Apply dropout to attention weights if specified
    let attn_weights = if dropout_p > 0.0 {
        // During training, randomly zero out elements with probability dropout_p
        // During inference (when requires_grad is false), scale by (1-dropout_p) for expectation preservation
        if training {
            // Create dropout mask: keep with probability (1-dropout_p), zero otherwise
            let dropout_mask = create_dropout_mask(&attn_weights, dropout_p as f32)?;
            // Apply mask and scale to maintain expected value
            attn_weights
                .mul(&dropout_mask)?
                .mul_scalar(T::from_f64(1.0 / (1.0 - dropout_p)).unwrap())?
        } else {
            // During inference, no dropout applied
            attn_weights
        }
    } else {
        attn_weights
    };

    // Apply attention weights to values: attn_weights @ V
    let output = attn_weights.matmul(&value_dense)?;

    Ok(output)
}

/// Applies softmax activation function along the last dimension.
///
/// Formula: `softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Tensor with softmax applied along the last dimension
pub fn softmax<
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T,
>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    T: DataType + FloatExt + Neg<Output = T> + PartialOrd + Clone + FromPrimitive,
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

        // Find maximum value for numerical stability
        let mut max_val = slice[0];
        for val in slice {
            if *val > max_val {
                max_val = *val;
            }
        }

        // Compute exp(x - max) and sum
        let mut exp_sum = T::from_f32(0.0).unwrap();
        let mut exp_values = Vec::with_capacity(last_dim_size);

        for val in slice {
            let shifted = *val - max_val;
            let exp_val = shifted.exp();
            exp_values.push(exp_val);
            exp_sum = exp_sum + exp_val;
        }

        // Normalize by sum
        for exp_val in exp_values {
            output_data.push(exp_val / exp_sum);
        }
    }

    Ok(Tensor::from_vec(output_data, input_shape)?)
}
