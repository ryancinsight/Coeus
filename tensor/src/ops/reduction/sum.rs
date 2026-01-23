//! Sum reduction operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::SumFunction;

/// Computes the sum of elements along specified dimensions.
pub fn sum<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + 'static + core::ops::Add<Output = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let mut result = tensor.sum_generic(dims, keepdim)?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = SumFunction::new(Arc::new(tensor.clone()), tensor.shape().dims().to_vec());
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

