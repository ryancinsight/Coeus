//! Element-wise exponential

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::ExpFunction;

/// Element-wise exponential
pub fn exp<
    T,
    B,
    S,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + 'static,
{
    let data = tensor.as_slice().iter().map(|&x| x.exp()).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = ExpFunction::new(Arc::new(tensor.clone()), Arc::new(result.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
