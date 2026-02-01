//! Max reduction operation

use crate::functions::MaxFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Computes the maximum of elements along specified dimensions.
pub fn max<B, T, S>(tensor: &Tensor<B, S, T>, dim: usize, keepdim: bool) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + 'static + FloatExt,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + Clone + 'static,
{
    let dims = [dim];
    let mut result = tensor.reduce_generic(
        Some(&dims),
        keepdim,
        |acc, x| if x > acc { x } else { acc },
        T::neg_infinity(),
    )?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        // Create mask for backward pass: 1 where x == max, 0 otherwise
        let res_broadcast = if keepdim {
            result.clone()
        } else {
            let mut new_shape = result.shape().dims().to_vec();
            new_shape.insert(dim, 1);
            let data = result.as_slice().to_vec();
            Tensor::from_vec_with_backend(data, &new_shape, result.backend().clone())?
        };

        let target_shape = tensor.shape().dims();
        let res_expanded = crate::ops::shape::broadcast_tensor_data(
            res_broadcast.as_slice(),
            res_broadcast.shape().dims(),
            target_shape,
        )?;

        let mask_data: Vec<T> = tensor
            .as_slice()
            .iter()
            .zip(res_expanded)
            .map(|(&x, m)| if x == m { T::one() } else { T::zero() })
            .collect();
        let mask =
            Tensor::from_vec_with_backend(mask_data, target_shape, tensor.backend().clone())?;

        let grad_fn = MaxFunction::new(Arc::new(tensor.clone()), Arc::new(mask), dim, keepdim);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
