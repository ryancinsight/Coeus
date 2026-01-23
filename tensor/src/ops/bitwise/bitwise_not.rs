//! Element-wise bitwise NOT operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use std::ops::Not;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Element-wise bitwise NOT operation on a tensor
pub fn bitwise_not<
    T: DataType + Not<Output = T> + Clone + 'static,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data: Vec<T> = tensor.as_slice().iter().map(|x| !x.clone()).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
