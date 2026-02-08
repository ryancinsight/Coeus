//! Mean reduction operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;

use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Computes the mean of elements along specified dimensions.
pub fn mean<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::dispatch::TensorStorageOps<T> + Clone + 'static,
{
    let sum_t = tensor.sum_generic(dims, keepdim)?;

    let numel: usize = tensor.shape().dims().iter().product();
    let out_numel: usize = sum_t.shape().dims().iter().product();
    let n = numel / out_numel;
    let factor = T::from_f64(1.0 / n as f64).unwrap();

    let mut result = sum_t.mul_scalar(factor)?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = crate::functions::MeanFunction::new(
            Arc::new(tensor.clone()),
            tensor.shape().dims().to_vec(),
        );
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
