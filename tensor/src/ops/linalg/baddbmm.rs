//! Batch add matrix multiplication.

use crate::ops::linalg::bmm;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Batch add matrix multiplication: beta * self + alpha * (batch1 @ batch2)
pub fn baddbmm<B, T, S>(
    input: &Tensor<B, S, T>,
    batch1: &Tensor<B, S, T>,
    batch2: &Tensor<B, S, T>,
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
    // C = beta * input + alpha * (batch1 @ batch2)
    
    let prod = bmm(batch1, batch2)?;
    let scaled_prod = crate::ops::arithmetic::mul(&prod, &Tensor::full_like(&prod, alpha)?)?;

    let scaled_input = crate::ops::arithmetic::mul(input, &Tensor::full_like(input, beta)?)?;

    crate::ops::arithmetic::add(&scaled_input, &scaled_prod)
}
