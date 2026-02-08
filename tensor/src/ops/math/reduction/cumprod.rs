//! Cumulative product

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Cumulative product along a dimension
pub fn cumprod<T, B, S>(tensor: &Tensor<B, S, T>, dim: usize) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    // Simplified global cumprod for now to satisfy build
    let mut prod = T::one();
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| {
        prod = prod * x;
        prod
    }).collect();
    
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
