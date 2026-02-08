//! Top-k operations

use crate::ops::TensorStorageOps;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Returns the k largest elements of the given input tensor along a given dimension.
///
/// Returns a tuple of (top_k_values_tensor, raw_indices_vec).
/// The caller is responsible for converting indices to a Tensor if needed.
pub fn topk<
    T: DataType + PartialOrd,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T>
        + Clone
        + Send
        + Sync
        + StorageFromVec<T>
        + storage::StorageToDense<T>
        + TensorStorageOps<T>
        + 'static,
>(
    tensor: &Tensor<B, S, T>,
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<(Tensor<B, S, T>, Vec<usize>)> {
    let dims = tensor.shape().dims();
    if dim >= dims.len() {
        return Err(crate::TensorError::InvalidDimension {
            dim,
            ndim: dims.len(),
        });
    }

    // Convert to dense for now
    let dense = tensor.to_dense_generic()?;
    let data = dense.as_slice();

    let mut pairs: Vec<(T, usize)> = data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    if largest {
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    } else {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    }

    let top_pairs = &pairs[..k.min(pairs.len())];

    let top_values: Vec<T> = top_pairs.iter().map(|p| p.0).collect();
    let top_indices: Vec<usize> = top_pairs.iter().map(|p| p.1).collect();

    let mut new_dims = dims.to_vec();
    new_dims[dim] = k.min(dims[dim]);

    let values_tensor =
        Tensor::from_vec_with_backend(top_values, &new_dims, tensor.backend.clone())?;

    Ok((values_tensor, top_indices))
}
