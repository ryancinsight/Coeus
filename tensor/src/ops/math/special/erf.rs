//! Element-wise error function

use crate::functions::math::ErfFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Element-wise error function
pub fn erf<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    // Use an approximation or libm if available. For now, we'll assume a direct call if supported by T.
    // In coeus, erf is often implemented via a custom trait or external crate.
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x.erf()).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = ErfFunction::new(Arc::new(tensor.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
