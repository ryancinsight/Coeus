//! Distance operations between tensors
//!
//! Implementation of pairwise distance and cosine similarity.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Computes the batchwise pairwise distance between vectors.
/// 
/// dist = || x1 - x2 + eps ||_p
pub fn pairwise_distance<T, B, S>(
    x1: &Tensor<B, S, T>,
    x2: &Tensor<B, S, T>,
    p: f64,
    eps: f64,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + num_traits::FromPrimitive + num_traits::Signed + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let diff = crate::ops::arithmetic::sub(x1, x2)?;
    
    // Add eps if necessary. PyTorch adds eps inside the norm.
    // dist = (sum(|x1 - x2|^p) + eps)^(1/p) is often used for stability.
    // Actually PyTorch says: || x1 - x2 ||_p with eps for stability.
    
    let eps_t = T::from_f64(eps).unwrap();
    
    // Compositional approach
    let abs_diff = crate::ops::math::abs(&diff)?;
    
    if p == 2.0 {
        let squared = crate::ops::math::square(&abs_diff)?;
        // sum over last dimension (usually)
        let rank = x1.shape().ndim();
        let dims = if rank > 0 { vec![rank - 1] } else { vec![] };
        let summed = crate::ops::reduction::sum(&squared, Some(&dims), keepdim)?;
        
        // Add eps stability
        let eps_tensor = Tensor::full_like(&summed, eps_t)?;
        let stabilized = crate::ops::arithmetic::add(&summed, &eps_tensor)?;
        crate::ops::math::sqrt(&stabilized)
    } else {
        let p_t = T::from_f64(p).unwrap();
        let inv_p = T::from_f64(1.0 / p).unwrap();
        let powered = crate::ops::math::pow_scalar(&abs_diff, p_t)?;
        let rank = x1.shape().ndim();
        let dims = if rank > 0 { vec![rank - 1] } else { vec![] };
        let summed = crate::ops::reduction::sum(&powered, Some(&dims), keepdim)?;
        
        // Add eps stability
        let eps_tensor = Tensor::full_like(&summed, eps_t)?;
        let stabilized = crate::ops::arithmetic::add(&summed, &eps_tensor)?;
        crate::ops::math::pow_scalar(&stabilized, inv_p)
    }
}

/// Computes the cosine similarity between x1 and x2 along dim.
/// 
/// sim = (x1 . x2) / (||x1||_2 * ||x2||_2 + eps)
pub fn cosine_similarity<T, B, S>(
    x1: &Tensor<B, S, T>,
    x2: &Tensor<B, S, T>,
    dim: usize,
    eps: f64,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + num_traits::FromPrimitive + num_traits::Signed + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let dot = crate::ops::arithmetic::mul(x1, x2)?;
    let dims = [dim];
    let dot_sum = crate::ops::reduction::sum(&dot, Some(&dims), false)?;
    
    let norm1 = crate::ops::math::norm(x1, Some(2.0), Some(&dims), false)?;
    let norm2 = crate::ops::math::norm(x2, Some(2.0), Some(&dims), false)?;
    
    let norms = crate::ops::arithmetic::mul(&norm1, &norm2)?;
    let eps_t = T::from_f64(eps).unwrap();
    
    // Instead of full_like, just add scalar if possible, or use full_like
    let eps_tensor: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![eps_t; norms.shape().size()], norms.shape().dims(), norms.backend().clone())?;
    let denom = crate::ops::arithmetic::add(&norms, &eps_tensor)?;
    
    crate::ops::arithmetic::div(&dot_sum, &denom)
}
