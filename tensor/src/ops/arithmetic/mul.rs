//! Mul operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise multiplication with broadcasting.
pub fn mul<
    T: DataType
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + Clone
        + Copy
        + num_traits::Zero
        + Default,
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
    super::broadcast_binary_op(a, b, |x, y| x * y)
}
