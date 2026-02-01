//! SVD decomposition operation

use crate::{Tensor, Result};
use storage::DenseStorage;
use backend::Backend;
use dtype::DataType;

/// Compute the Shingular Value Decomposition (SVD) of a matrix or a batch of matrices.
/// A = U S V^T
pub fn svd<B: Backend, S, T>(tensor: &Tensor<B, S, T>) -> Result<(Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>)>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    let backend = tensor.backend();
    let storage = tensor.storage();
    
    let dense_storage = storage.storage_to_dense()?;
    
    let (u_storage, s_storage, vt_storage) = backend.svd_dense(&dense_storage)?;
    
    let u = Tensor::from_storage(u_storage, backend.clone());
    let s = Tensor::from_storage(s_storage, backend.clone());
    let vt = Tensor::from_storage(vt_storage, backend.clone());
    
    Ok((u, s, vt))
}

impl<B: Backend, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    /// Compute the SVD decomposition of a matrix
    pub fn svd(&self) -> Result<(Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>)> {
        svd(self)
    }
}
