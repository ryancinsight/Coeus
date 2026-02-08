//! Element-wise log(1 + x)

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise log(1 + x)
pub fn log1p<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = tensor.storage.storage_log1p(&tensor.backend)?;
    Ok(Tensor::from_storage(storage, tensor.backend.clone()))
}
