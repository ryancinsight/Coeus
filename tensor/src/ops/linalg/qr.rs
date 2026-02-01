//! QR decomposition operation

use crate::{Tensor, Result};
use storage::DenseStorage;
use backend::Backend;
use dtype::DataType;

/// Compute the QR decomposition of a matrix or a batch of matrices.
/// A = QR
pub fn qr<B: Backend, S, T>(tensor: &Tensor<B, S, T>) -> Result<(Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>)>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    let backend = tensor.backend();
    let storage = tensor.storage();
    
    let dense_storage = storage.storage_to_dense()?;
    
    let (q_storage, r_storage) = backend.qr_dense(&dense_storage)?;
    
    let q = Tensor::from_storage(q_storage, backend.clone());
    let r = Tensor::from_storage(r_storage, backend.clone());
    
    Ok((q, r))
}

impl<B: Backend, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: crate::ops::dispatch::traits::TensorStorageOps<T>,
    T: DataType + num_traits::Float + Default,
{
    /// Compute the QR decomposition of a matrix
    pub fn qr(&self) -> Result<(Tensor<B, DenseStorage<T>, T>, Tensor<B, DenseStorage<T>, T>)> {
        qr(self)
    }
}
