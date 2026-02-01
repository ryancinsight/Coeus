use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise copysign: returns magnitude(input) * sign(other)
pub fn copysign<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    input: &Tensor<B, S, T>,
    other: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = input
        .as_slice()
        .iter()
        .zip(other.as_slice())
        .map(|(&i, &o)| i.abs().copysign(o))
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
