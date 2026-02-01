//! Add operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise addition with broadcasting.
pub fn add<
    T: DataType + std::ops::Add<Output = T> + Clone + Copy,
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
    // If different storage, convert to dense for safety for now until mixed dispatch is impl
    // Or check TypeId?
    // For now, fallback to dense broadcast which handles conversion.
    // Optimization: if S1==S2 and shape matches, use storage_add.
    // We can't easily check S1==S2 at runtime with generics without specialization or Any.
    // But we can try to use broadcast_binary_op unconditionally if we want safe mixed support.
    // OR: rely on compiler monomorphization.
    
    // Simplest safe approach:
    // If shapes match:
    //    Convert to dense -> add -> return
    //    OR: if S1, S2 are same (how to check?), call storage_add.
    
    // Given the constraints and the need for correct mixed ops today:
    // We will use broadcast_binary_op which converts to dense.
    // Ideally we optimize the S1==S2 case.
    // We can rely on to_dense being no-op if already dense? No, it clones.
    
    // Let's rely on broadcast_binary_op for robustness for now.
    // It calls conversion.
    super::broadcast_binary_op(a, b, |x, y| x + y)

    // Note: This disables the specialized storage_add path.
    // To re-enable it:
    // We would need to know S1 == S2.
    // Since we can't easily, we might lose performance on Dense+Dense unless broadcast_binary_op is fast.
    // broadcast_binary_op allocates vectors.
}
