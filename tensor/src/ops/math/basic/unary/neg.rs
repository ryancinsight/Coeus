//! Element-wise negation

use crate::functions::math::NegFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Signed;
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise negation
pub fn neg<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Signed + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| -x).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = NegFunction::new(Arc::new(tensor.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
