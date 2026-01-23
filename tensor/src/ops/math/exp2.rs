//! Element-wise base-2 exponential (2^x)

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Element-wise base-2 exponential: 2^x
pub fn exp2<
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|&x| x.exp2()).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
