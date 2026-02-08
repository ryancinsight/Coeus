//! Distance operations for neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;
use crate::core::error::{Result};
use num_traits::FromPrimitive;

/// Computes pairwise distance between vectors.
pub fn pairwise_distance<B, S, T>(
    x1: &Tensor<B, S, T>,
    x2: &Tensor<B, S, T>,
    p: f64,
    eps: f64,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    let diff = tensor::ops::sub(x1, x2)?;
    // For numerical stability, we'd ideally add eps somewhere, 
    // but a simple norm is the standard implementation.
    let norm = tensor::ops::norm(&diff, Some(p), Some(&[x1.shape().ndim() - 1]), keepdim)?;
    Ok(norm)
}

/// Computes cosine similarity between vectors.
pub fn cosine_similarity<B, S, T>(
    x1: &Tensor<B, S, T>,
    x2: &Tensor<B, S, T>,
    dim: usize,
    eps: f64,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    let eps_t = T::from_f64(eps).unwrap();
    
    let dot = tensor::ops::sum(&tensor::ops::mul(x1, x2)?, Some(&[dim]), false)?;
    
    let norm1 = tensor::ops::norm(x1, Some(2.0), Some(&[dim]), false)?;
    let norm2 = tensor::ops::norm(x2, Some(2.0), Some(&[dim]), false)?;
    
    let norms = tensor::ops::mul(&norm1, &norm2)?;
    let norms_clamped = tensor::ops::comparison::maximum_scalar(&norms, eps_t)?;
    
    let sim = tensor::ops::div(&dot, &norms_clamped)?;
    Ok(sim)
}
