//! Element-wise complementary error function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::{FromPrimitive, ToPrimitive};
use storage::{Storage, StorageFromVec};

/// Element-wise complementary error function
pub fn erfc<
    T: DataType + num_traits::Float + 'static,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + storage::StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = tensor.storage.storage_erfc(&tensor.backend)?;
    let mut result = Tensor::from_storage(storage, tensor.backend.clone());

    if tensor.requires_grad {
        result = result.requires_grad_(true);
    }

    Ok(result)
}
