//! Sorting operations

use crate::{Result, Tensor};
use crate::ops::TensorStorageOps;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Sorts the elements of the input tensor along a given dimension.
///
/// Returns a tuple of (sorted_values_tensor, raw_indices_vec).
/// The caller is responsible for converting indices to a Tensor if needed.
pub fn sort<
    T: DataType + PartialOrd,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
    descending: bool,
) -> Result<(Tensor<B, S, T>, Vec<usize>)> {
    let dims = tensor.shape().dims();
    if dim >= dims.len() {
        return Err(crate::TensorError::InvalidDimension {
            dim,
            ndim: dims.len(),
        });
    }

    // Convert to dense for sorting
    let dense = tensor.to_dense_generic()?;
    let data = dense.as_slice();
    
    // Create pairs of (value, index)
    let mut pairs: Vec<(T, usize)> = data.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    
    if descending {
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    } else {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    }

    let sorted_values: Vec<T> = pairs.iter().map(|p| p.0).collect();
    let sorted_indices: Vec<usize> = pairs.iter().map(|p| p.1).collect();

    let values_tensor = Tensor::from_vec_with_backend(sorted_values, dims, tensor.backend.clone())?;

    Ok((values_tensor, sorted_indices))
}
