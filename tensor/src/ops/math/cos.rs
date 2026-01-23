//! Element-wise cosine

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise cosine
pub fn cos<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + 'static,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x.cos()).collect();
    let mut result = Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}
