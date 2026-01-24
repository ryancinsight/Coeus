//! Add matrix multiplication with scaling.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, traits::FloatExt};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::ops::linalg::matmul;

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
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
{
    // C = beta * input + alpha * (mat1 @ mat2)
    // Implicitly supports autograd via composition
    let prod = matmul(mat1, mat2)?;
    let scaled_prod = crate::ops::arithmetic::mul(
        &prod,
        &Tensor::full_like(&prod, alpha)?
    )?;
    
    let scaled_input = crate::ops::arithmetic::mul(
        input,
        &Tensor::full_like(input, beta)?
    )?;
    
    crate::ops::arithmetic::add(&scaled_input, &scaled_prod)
}
