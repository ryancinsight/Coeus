//! Neg operation

use crate::functions::NegFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise negation.
pub fn neg<
    T: DataType + std::ops::Neg<Output = T> + Clone + Copy,
    B: Backend<Data = T>
        + Clone
        + Send
        + Sync
        + Default
        + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>
        + 'static,
    S: Storage<T>
        + Clone
        + Send
        + Sync
        + StorageFromVec<T>
        + storage::StorageToDense<T>
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    // Delegate to storage implementation via unified OPS trait
    let result_storage = tensor.storage.storage_neg(&tensor.backend)?;
    let mut result = Tensor::from_storage(result_storage, tensor.backend.clone());

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = NegFunction::new(Arc::new(tensor.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
