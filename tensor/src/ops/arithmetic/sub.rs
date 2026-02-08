//! Sub operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise subtraction with broadcasting.
pub fn sub<
    T: DataType + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + Clone + Copy,
    B: Backend<Data = T>
        + Clone
        + Send
        + Sync
        + Default
        + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T>
        + 'static,
    S1: Storage<T>
        + Clone
        + Send
        + Sync
        + StorageFromVec<T>
        + storage::StorageToDense<T>
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
    S2: Storage<T>
        + Clone
        + Send
        + Sync
        + StorageFromVec<T>
        + storage::StorageToDense<T>
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
>(
    a: &Tensor<B, S1, T>,
    b: &Tensor<B, S2, T>,
) -> Result<Tensor<B, S1, T>> {
    let mut result = super::broadcast_binary_op(a, b, |x, y| x - y)?;

    if crate::tensor_core::grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        use crate::functions::SubFunction;
        use std::sync::Arc;

        let a_arc = Arc::new(a.clone());
        let b_dense = b.to_dense_generic()?;
        let b_s1 = Tensor::<B, S1, T>::from_vec_with_backend(
            b_dense.as_slice().to_vec(),
            b.shape().dims(),
            b.backend().clone()
        )?;
        
        let grad_fn = SubFunction::new(a_arc, Arc::new(b_s1));
        result = result.requires_grad_(true).with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
