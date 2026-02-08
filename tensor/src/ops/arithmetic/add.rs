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
    let mut result = super::broadcast_binary_op(a, b, |x, y| x + y)?;

    if crate::tensor_core::grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        // We only support autograd for same storage types or when we can convert safely.
        // For now, assume a and b can be promoted as used by AddFunction.
        // If S1 != S2, we would need to convert b to S1.
        // Since we are returning S1, we must have S1 in the grad_fn.
        
        // This is a bit of a hack if S1 != S2, but for same-type it's correct.
        // A truly robust solution would need mixed-input functions.
        
        // Safety: if TypeId doesn't match, this might fail at compile time if we try to force it,
        // but here it's fine as long as we can construct AddFunction.
        // But AddFunction<B, S1, T> expects Arc<Tensor<B, S1, T>>.
        // We can't easily convert Arc<Tensor<B, S2, T>> to Arc<Tensor<B, S1, T>> without data copy if they differ.
        
        // For the common case (S1 == S2), this is straightforward.
        // We'll use a helper or just try to be specific for Dense.
        
        // If they are both Dense, it's easy.
        use crate::functions::AddFunction;
        use std::sync::Arc;

        // Try to promote to Arc. If they are already the same type, this works.
        // If S2 != S1, this will fail to compile if we just use b.clone().
        // So we need a way to "cast" or "convert" b to S1.
        
        let a_arc = Arc::new(a.clone());
        // For now, if S2 is not S1, we'll try to convert.
        // This requires S2 to be convertible to S1. 
        // Since we don't have a direct trait for that, we'll just use Dense as a fallback
        // if we can't match S1.
        
        // Actually, let's just use DenseStorage for autograd if we are in doubt? 
        // No, result is S1.
        
        // Simplified for now: assume S1 == S2 or user handles conversion.
        // The test has S1 == S2 == Dense.
        
        // We'll use a trick: if S2 is not S1, we'll convert to dense then to S1 if possible?
        // Or just fail?
        
        // Let's settle for: if they match, we record. 
        // Since we can't easily check TypeId of S1/S2 here (no 'static),
        // we'll just try to use a.clone() and b.clone() and hope for the best in same-type scenarios.
        // If S1 != S2, this line will fail to compile if b.clone() is S2.
        
        // Wait! a.clone() is S1. b.clone() is S2.
        // AddFunction<B, S1, T> needs S1.
        // So we MUST convert b to S1.
        
        let b_dense = b.to_dense_generic()?;
        // Convert dense to S1
        let b_s1 = Tensor::<B, S1, T>::from_vec_with_backend(
            b_dense.as_slice().to_vec(),
            b.shape().dims(),
            b.backend().clone()
        )?;
        
        let grad_fn = AddFunction::new(a_arc, Arc::new(b_s1));
        result = result.requires_grad_(true).with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
