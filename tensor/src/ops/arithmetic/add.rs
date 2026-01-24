//! Add operation

use crate::functions::AddFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use std::sync::Arc;

/// Element-wise addition with broadcasting.
pub fn add<
    T: DataType + std::ops::Add<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::dispatch::TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let mut result = if a.shape() == b.shape() {
        // Delegate to storage implementation via unified OPS trait
        let result_storage = a.storage.storage_add(&b.storage, &a.backend)?;
        Tensor::from_storage(result_storage, a.backend.clone())
    } else {
        super::broadcast_binary_op(a, b, |x, y| x + y)?
    };

    if crate::tensor_core::grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = AddFunction::new(Arc::new(a.clone()), Arc::new(b.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
