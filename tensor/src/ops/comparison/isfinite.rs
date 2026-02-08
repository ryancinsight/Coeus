//! Element-wise is_finite check

use crate::{Result, Tensor, Backend, DataType, Storage};
use crate::ops::TensorStorageOps;

/// Returns a new tensor with boolean elements representing if each element is finite (not NaN, not Inf)
pub fn isfinite<
    T: DataType + num_traits::Float + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    input: &Tensor<B, S, T>,
) -> crate::Result<Tensor<B, S, T>> {
    let storage = input.storage.storage_isfinite(&input.backend)?;
    Ok(Tensor::from_storage(storage, input.backend.clone()))
}
