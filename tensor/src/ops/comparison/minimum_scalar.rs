//! Minimum with scalar operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise minimum of tensor and scalar
pub fn minimum_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialOrd + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x < scalar { x } else { scalar })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
