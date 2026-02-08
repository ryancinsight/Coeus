//! Element-wise power operations

use crate::functions::math::PowFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise power (tensor ** tensor)
pub fn pow<T, B, S>(input: &Tensor<B, S, T>, exp: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(exp.as_slice().iter()).map(|(&base, &e)| base.powf(e)).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())?;

    if crate::tensor_core::grad_enabled() && (input.requires_grad() || exp.requires_grad()) {
        let grad_fn = PowFunction::new(Arc::new(input.clone()), Arc::new(exp.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

/// Element-wise power (tensor ** scalar)
pub fn pow_scalar<T, B, S>(input: &Tensor<B, S, T>, exp: T) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().map(|&base| base.powf(exp)).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
