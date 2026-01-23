//! Degrees to radians

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Degrees to radians.
pub fn deg2rad<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(T::zero);
    let factor = pi / T::from(180.0).unwrap_or_else(T::one);
    let data = tensor.as_slice().iter().map(|&x| x * factor).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
