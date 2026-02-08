//! Non-zero element indexing.

use crate::{Result, Tensor, CpuBackend, DenseStorage};
use backend::Backend;
use dtype::{DataType, I64};
use storage::{Storage, StorageToDense, StorageFromVec};
use num_traits::Zero;

/// Returns a tensor containing the indices of all non-zero elements of input.
///
/// Each row in the result contains the indices of a non-zero element in input.
/// The result is a 2-D tensor of shape (N, ndim), where N is the number of non-zero elements.
pub fn nonzero<B, S, T>(tensor: &Tensor<B, S, T>) -> Result<Tensor<CpuBackend<I64>, DenseStorage<I64>, I64>>
where
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Zero + 'static,
{
    let dense = tensor.to_dense_generic()?;
    let data = dense.as_slice();
    let dims = tensor.storage.shape().dims();
    let ndim = dims.len();
    
    let mut flat_indices = Vec::new();
    let mut n_nonzero = 0;
    
    for (i, val) in data.iter().enumerate() {
        if !val.is_zero() {
            n_nonzero += 1;
            // Convert flat index to multi-dimensional indices
            let mut remaining = i;
            let mut coord = vec![0; ndim];
            for d in (0..ndim).rev() {
                coord[d] = remaining % dims[d];
                remaining /= dims[d];
            }
            for c in coord {
                flat_indices.push(I64(c as i64));
            }
        }
    }
    
    Tensor::from_vec_with_backend(
        flat_indices,
        &[n_nonzero, ndim],
        CpuBackend::new(),
    )
}
