//! Element-wise arctangent (inverse tangent)

use crate::functions::math::AtanFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, FromPrimitive};
use std::sync::Arc;
use storage::{Storage, StorageFromVec};

/// Element-wise arctangent (inverse tangent)
///
/// Returns values in radians in the range [-π/2, π/2].
pub fn atan<
    T: DataType + Float + dtype::traits::FloatExt + FromPrimitive,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x.atan()).collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad {
        let grad_fn = AtanFunction::new(Arc::new(tensor.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
