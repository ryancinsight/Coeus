use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise square
pub fn square<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|&x| x * x).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
