//! Reshape operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::ReshapeFunction;

/// Returns a tensor with the same data as the input but with a new shape.
pub fn reshape<B, T, S>(
    tensor: &Tensor<B, S, T>,
    shape: &[usize],
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let old_size: usize = tensor.shape().dims().iter().product();
    let new_size: usize = shape.iter().product();

    if old_size != new_size {
        return Err(TensorError::ShapeMismatch {
            expected: tensor.shape().dims().to_vec(),
            actual: shape.to_vec(),
            operation: "reshape",
        });
    }

    let mut result = Tensor::from_vec_with_backend(tensor.as_slice().to_vec(), shape, tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = ReshapeFunction::new(Arc::new(tensor.clone()), tensor.shape().dims().to_vec());
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
