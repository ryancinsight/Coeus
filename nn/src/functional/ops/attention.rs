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

use crate::core::error::{NNError, Result};

fn create_dropout_mask<B, S, T>(tensor: &Tensor<B, S, T>, dropout_p: f32) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + FromPrimitive,
{
    use rand::prelude::*;

    let shape = tensor.shape();
    let size = shape.dims().iter().product();

    let mut rng = rand::thread_rng();
    let mut mask_data = Vec::with_capacity(size);

    for _ in 0..size {
        let keep = rng.gen::<f32>() > dropout_p;
        mask_data.push(if keep {
            T::from_f32(1.0).unwrap()
        } else {
            T::from_f32(0.0).unwrap()
        });
    }

    Ok(Tensor::from_vec_with_backend(
        mask_data,
        shape.dims(),
        tensor.backend().clone(),
    )?)
}

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

    let query_dense = query.to_dense_generic()?;
    let key_dense = key.to_dense_generic()?;
    let value_dense = value.to_dense_generic()?;

    let key_t = key_dense.transpose(key_shape.len() - 2, key_shape.len() - 1)?;
    let attn_scores = tensor::ops::matmul(&query_dense, &key_t)?;

    let scale_factor = T::from_f64((d_k as f64).sqrt()).unwrap();
    let attn_scores_scaled = attn_scores.mul_scalar(T::from_f64(1.0).unwrap() / scale_factor)?;

    let attn_scores = if let Some(mask) = attn_mask {
        let mask_dense = mask.to_dense_generic()?;
        tensor::ops::add(&attn_scores_scaled, &mask_dense)?
    } else {
        attn_scores_scaled
    };

    let attn_weights = crate::ops::activation::softmax(&attn_scores)?;

    let attn_weights = if dropout_p > 0.0 {
        if training {
            let dropout_mask = create_dropout_mask(&attn_weights, dropout_p as f32)?;
            tensor::ops::mul(&attn_weights, &dropout_mask)?
                .mul_scalar(T::from_f64(1.0 / (1.0 - dropout_p)).unwrap())?
        } else {
            attn_weights
        }
    } else {
        attn_weights
    };

    let output = tensor::ops::matmul(&attn_weights, &value_dense)?;

    Ok(output)
}
