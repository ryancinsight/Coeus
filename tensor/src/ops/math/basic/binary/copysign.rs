//! Element-wise copysign

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise copysign
pub fn copysign<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&mag, &sign)| mag.copysign(sign)).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
