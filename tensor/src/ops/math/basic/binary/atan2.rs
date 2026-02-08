//! Element-wise atan2

use crate::functions::math::Atan2Function;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise atan2
pub fn atan2<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&y, &x)| y.atan2(x)).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())?;

    if crate::tensor_core::grad_enabled() && (input.requires_grad() || other.requires_grad()) {
        let grad_fn = Atan2Function::new(Arc::new(input.clone()), Arc::new(other.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
