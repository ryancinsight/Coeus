use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::Storage;

/// Selects a slice along a dimension.
///
/// Returns a tensor which skips the `index` in `dim`.
/// Practically, this is equivalent to `tensor.narrow(dim, index, 1).squeeze(dim)`.
///
/// # Arguments
/// * `tensor` - The input tensor.
/// * `dim` - The dimension to slice.
/// * `index` - The index to select.
///
/// # Returns
/// A new tensor with the selected slice.
pub fn select<B, S, T>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
    index: usize,
) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T> + crate::ops::TensorStorageOps<T>,
    T: DataType + core::ops::Neg<Output = T>,
{
    let narrowed = tensor.narrow(dim, index, 1)?;
    crate::ops::shape::squeeze::squeeze(&narrowed, dim)
}
