//! Outer product of two vectors.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};

/// Add outer product of vectors to input.
/// result = beta * input + alpha * (vec1 @ vec2)
pub fn addr<B, T, S>(
    input: &Tensor<B, S, T>,
    vec1: &Tensor<B, S, T>,
    vec2: &Tensor<B, S, T>,
    beta: T,
    alpha: T,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType
        + FloatExt
        + Clone
        + Copy
        + num_traits::Zero
        + num_traits::FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + Clone + Send + Sync + 'static,
{
    // outer product
    let outer_prod = super::outer::outer(vec1, vec2)?; 

    // Implicitly supports autograd via composition
    // We need to match S type. outer returns DenseStorage.
    // Ideally we convert outer_prod to S if possible, or everything to Dense.
    // For now let's rely on standard ops handling mixed storage or converting.
    // Actually standard ops usually return S1 (left operand storage).
    
    // alpha * outer
    let scaled_outer = crate::ops::arithmetic::mul(&outer_prod, &Tensor::full_like(&outer_prod, alpha)?)?;
    
    // beta * input
    let scaled_input = crate::ops::arithmetic::mul(input, &Tensor::full_like(input, beta)?)?;

    // result = scaled_input + scaled_outer
    // This might require casting scaled_outer to S if S is not DenseStorage.
    // But arithmetic ops usually handle this via delegation or conversion.
    // Let's assume input's storage S is the target.
    
    // We need to make sure shapes match for add.
    // input must bebroadcastable to [M, N].
    
    crate::ops::arithmetic::add(&scaled_input, &scaled_outer)
}
