//! Add matrix-vector multiplication.

use crate::ops::linalg::{mv, matmul}; // Fallback uses mv or matmul? mv.
use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Add matrix-vector multiplication: beta * self + alpha * (mat @ vec)
pub fn addmv<B, T, S>(
    input: &Tensor<B, S, T>,
    mat: &Tensor<B, S, T>,
    vec: &Tensor<B, S, T>,
    beta: T,
    alpha: T,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Clone + Copy + 'static,
    S: Storage<T>
        + StorageToDense<T>
        + StorageFromVec<T>
        + Clone
        + Send
        + Sync
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
{
    // Try optimized dispatch first
    /*
    match input.storage().storage_addmv(
        mat.storage(),
        vec.storage(),
        beta,
        alpha,
        input.backend(),
    ) {
        Ok(storage) => return Ok(Tensor::from_storage(storage, input.backend().clone())),
        // Fallback if not supported by storage or backend
        Err(crate::TensorError::UnsupportedOperation { .. })
        | Err(crate::TensorError::BackendUnsupported { .. }) => {}
        // Propagate other errors
        Err(e) => return Err(e),
    }
    */

    // Default composition logic
    let prod = mv(mat, vec)?;
    let scaled_prod = crate::ops::arithmetic::mul(&prod, &Tensor::full_like(&prod, alpha)?)?;
    
    let scaled_input = crate::ops::arithmetic::mul(input, &Tensor::full_like(input, beta)?)?;
    
    crate::ops::arithmetic::add(&scaled_input, &scaled_prod)
}
