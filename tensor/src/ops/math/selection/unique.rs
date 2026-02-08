//! Unique operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Returns the unique elements of the input tensor.
pub fn unique<
    T: DataType + PartialOrd,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let dense = tensor.to_dense_generic()?;
    let data = dense.as_slice();

    let mut unique_values: Vec<T> = data.iter().copied().collect();
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    unique_values.dedup();

    let n = unique_values.len();
    Tensor::from_vec_with_backend(unique_values, &[n], tensor.backend.clone())
}
