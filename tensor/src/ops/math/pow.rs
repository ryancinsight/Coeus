//! Element-wise power

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, Num};
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise power
pub fn pow<
    T: DataType + Float + Num + Clone + dtype::traits::FloatExt,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    base: &Tensor<B, S, T>,
    exponent: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if base.shape() != exponent.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: base.shape().dims().to_vec(),
            actual: exponent.shape().dims().to_vec(),
            operation: "pow",
        });
    }

    let data = base
        .as_slice()
        .iter()
        .zip(exponent.as_slice())
        .map(|(&b, &e)| {
            if b < T::zero() && T::from(2.0).is_some_and(|two| e % two != T::zero()) {
                T::nan()
            } else if b == T::zero() && e < T::zero() {
                T::infinity()
            } else {
                b.powf(e)
            }
        })
        .collect();

    let mut result =
        Tensor::from_vec_with_backend(data, base.shape().dims(), base.backend.clone())?;

    if crate::tensor_core::grad_enabled() && (base.requires_grad() || exponent.requires_grad()) {
        let grad_fn = crate::functions::math::PowBinaryFunction::new(
            Arc::new(base.clone()),
            Arc::new(exponent.clone()),
        );
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

/// Element-wise power with scalar exponent
pub fn pow_scalar<
    T: DataType + Float + Num + Clone + dtype::traits::FloatExt,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    exponent: T,
) -> Result<Tensor<B, S, T>> {
    let data = tensor
        .as_slice()
        .iter()
        .map(|&b| {
            if b < T::zero() && T::from(2.0).is_some_and(|two| exponent % two != T::zero()) {
                T::nan()
            } else if b == T::zero() && exponent < T::zero() {
                T::infinity()
            } else {
                b.powf(exponent)
            }
        })
        .collect();

    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = crate::functions::math::PowFunction::new(Arc::new(tensor.clone()), exponent);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
