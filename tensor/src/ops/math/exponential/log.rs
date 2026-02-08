//! Element-wise natural logarithm

use crate::functions::LogFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use num_traits::Float;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Element-wise natural logarithm
pub fn log<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    T: DataType + Float + FloatExt + 'static,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data = tensor.as_slice().iter().map(|&x| x.ln()).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = LogFunction::new(Arc::new(tensor.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
