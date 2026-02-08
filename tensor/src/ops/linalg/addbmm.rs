//! Add batch matrix multiplication (summed).

use crate::ops::linalg::bmm;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Add batch matrix multiplication: beta * self + alpha * (batch1 @ batch2).sum(0)
pub fn addbmm<B, T, S>(
    input: &Tensor<B, S, T>,
    batch1: &Tensor<B, S, T>,
    batch2: &Tensor<B, S, T>,
    beta: T,
    alpha: T,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Clone + Copy + std::ops::Add<Output = T> + 'static,
    S: Storage<T>
        + StorageToDense<T>
        + StorageFromVec<T>
        + Clone
        + Send
        + Sync
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
{
    // C = beta * input + alpha * (sum(batch1 @ batch2))
    
    let prod = bmm(batch1, batch2)?;
    // Sum along batch dimension (dim 0)
    // Reduce to [N, P] from [B, N, P]
    let summed_prod = crate::ops::reduction::sum(&prod, Some(&[0]), false)?; // keepdim=false
    
    let scaled_prod = crate::ops::arithmetic::mul(&summed_prod, &Tensor::full_like(&summed_prod, alpha)?)?;

    let scaled_input = crate::ops::arithmetic::mul(input, &Tensor::full_like(input, beta)?)?;

    crate::ops::arithmetic::add(&scaled_input, &scaled_prod)
}
