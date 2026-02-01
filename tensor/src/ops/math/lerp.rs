use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Linear interpolation: input + weight * (end - input)
pub fn lerp<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    input: &Tensor<B, S, T>,
    end: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if input.shape() != end.shape() || input.shape() != weight.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: input.shape().dims().to_vec(),
            actual: end.shape().dims().to_vec(),
            operation: "lerp",
        });
    }
    let data = input
        .as_slice()
        .iter()
        .zip(end.as_slice())
        .zip(weight.as_slice())
        .map(|((&i, &e), &w)| i + w * (e - i))
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

/// Linear interpolation with scalar weight
pub fn lerp_scalar<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    input: &Tensor<B, S, T>,
    end: &Tensor<B, S, T>,
    weight: T,
) -> Result<Tensor<B, S, T>> {
    if input.shape() != end.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: input.shape().dims().to_vec(),
            actual: end.shape().dims().to_vec(),
            operation: "lerp_scalar",
        });
    }
    let data = input
        .as_slice()
        .iter()
        .zip(end.as_slice())
        .map(|(&i, &e)| i + weight * (e - i))
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
