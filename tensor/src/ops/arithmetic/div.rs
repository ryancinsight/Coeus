//! Div operation

use crate::functions::DivFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use std::sync::Arc;

/// Element-wise division with broadcasting.
pub fn div<
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Div<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Neg<Output = T>,
    B: Backend<Data = T> + Clone + Send + Sync + Default + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::dispatch::TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let mut result = if a.shape() == b.shape() {
        // Explicitly use storage operation for dispatch
        let result_storage = a.storage.storage_div(&b.storage, &a.backend)?;
        Tensor::from_storage(result_storage, a.backend.clone())
    } else {
        super::broadcast_binary_op(a, b, |x, y| x / y)?
    };

    if crate::tensor_core::grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = DivFunction::new(Arc::new(a.clone()), Arc::new(b.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
