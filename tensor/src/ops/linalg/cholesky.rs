//! Cholesky decomposition operation

use crate::{Tensor, Result};
use storage::DenseStorage;
use backend::Backend;
use dtype::DataType;

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix
/// or a batch of such matrices.
/// A = L L^T
pub fn cholesky<B: Backend, S, T>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    let backend = tensor.backend();
    let storage = tensor.storage();
    
    // For now, only support dense matrices (CPU backend implementation already exists)
    // If strided, it will be converted to dense by the dispatch layer or here
    let dense_storage = storage.storage_to_dense()?;
    
    let res_storage = backend.cholesky_dense(&dense_storage)?;
    
    Ok(Tensor::from_storage(res_storage, backend.clone()))
}

impl<B: Backend, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    /// Compute the Cholesky decomposition of a symmetric positive-definite matrix
    pub fn cholesky(&self) -> Result<Tensor<B, DenseStorage<T>, T>> {
        cholesky(self)
    }
}
