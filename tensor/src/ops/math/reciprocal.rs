//! Element-wise reciprocal (1/x)

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise reciprocal (1/x)
pub fn reciprocal<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|&x| T::one() / x).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
