//! Add matrix multiplication with scaling.

use crate::ops::linalg::matmul;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Add matrix multiplication: beta * self + alpha * (mat1 @ mat2)
pub fn addmm<B, T, S>(
    input: &Tensor<B, S, T>,
    mat1: &Tensor<B, S, T>,
    mat2: &Tensor<B, S, T>,
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
    match input.storage().storage_addmm(
        mat1.storage(),
        mat2.storage(),
        beta,
        alpha,
        input.backend(),
    ) {
        Ok(storage) => return Ok(Tensor::from_storage(storage, input.backend().clone())),
        // Fallback if not supported by storage or backend
        Err(crate::TensorError::UnsupportedOperation { .. })
        | Err(crate::TensorError::BackendUnsupported { .. }) => {}
        Err(e) => return Err(e),
    }

    // C = beta * input + alpha * (mat1 @ mat2)
    // Implicitly supports autograd via composition
    let prod = matmul(mat1, mat2)?;
    let scaled_prod = crate::ops::arithmetic::mul(&prod, &Tensor::full_like(&prod, alpha)?)?;

    let scaled_input = crate::ops::arithmetic::mul(input, &Tensor::full_like(input, beta)?)?;

    crate::ops::arithmetic::add(&scaled_input, &scaled_prod)
}
