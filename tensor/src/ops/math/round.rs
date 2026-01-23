//! Element-wise round

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise rounding to nearest integer
pub fn round<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x.round()).collect();
    let mut result = Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}
