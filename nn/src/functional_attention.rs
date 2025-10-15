//! Attention mechanisms for neural networks.
//!
//! This module provides stateless attention operations including
//! scaled dot-product attention for transformer architectures.

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};

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
/// use coeus_nn::functional_attention::scaled_dot_product_attention;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let batch_size = 2;
/// let seq_len = 10;
/// let d_k = 64;
/// let d_v = 64;
///
/// let query = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_k]).unwrap();
/// let key = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_k]).unwrap();
/// let value = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::randn(&[batch_size, seq_len, d_v]).unwrap();
///
/// let output = scaled_dot_product_attention(&query, &key, &value, None, 0.0).unwrap();
/// assert_eq!(output.shape().dims(), &[batch_size, seq_len, d_v]);
/// ```
pub fn scaled_dot_product_attention<B, S, T>(
    query: &Tensor<B, S, T>,
    key: &Tensor<B, S, T>,
    value: &Tensor<B, S, T>,
    attn_mask: Option<&Tensor<B, S, T>>,
    dropout_p: f64,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + PartialOrd,
{
    let query_shape = query.shape().dims();
    let key_shape = key.shape().dims();
    let value_shape = value.shape().dims();

    if query_shape.len() < 2 || key_shape.len() < 2 || value_shape.len() < 2 {
        return Err(NNError::InvalidInput {
            message: "Query, key, and value must have at least 2 dimensions".to_string(),
        });
    }

    let query_seq_len = query_shape[query_shape.len() - 2];
    let key_seq_len = key_shape[key_shape.len() - 2];
    let value_seq_len = value_shape[value_shape.len() - 2];
    let d_k = *query_shape.last().unwrap();

    if *key_shape.last().unwrap() != d_k {
        return Err(NNError::InvalidInput {
            message: format!(
                "Query and key last dimensions must match, got {} and {}",
                d_k, key_shape.last().unwrap()
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
    let scale_factor = T::from((d_k as f64).sqrt()).unwrap();
    let attn_scores_scaled = attn_scores / scale_factor;

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

    // Apply dropout if specified (placeholder for now)
    let _dropout_p = dropout_p;

    // Apply attention weights to values: attn_weights @ V
    let output = attn_weights.matmul(&value_dense)?;

    output.to_generic()
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
pub fn softmax<T: DataType + FloatExt + std::ops::Neg<Output = T> + PartialOrd>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
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
        let mut max_val = slice[0].clone();
        for val in slice {
            if *val > max_val {
                max_val = val.clone();
            }
        }

        // Compute exp(x - max) and sum
        let mut exp_sum = T::from(0.0).unwrap();
        let mut exp_values = Vec::with_capacity(last_dim_size);

        for val in slice {
            let shifted = val.clone() - max_val.clone();
            let exp_val = shifted.exp();
            exp_values.push(exp_val.clone());
            exp_sum = exp_sum + exp_val;
        }

        // Normalize by sum
        for exp_val in exp_values {
            output_data.push(exp_val / exp_sum.clone());
        }
    }

    Tensor::from_vec(output_data, &input_shape)
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}
