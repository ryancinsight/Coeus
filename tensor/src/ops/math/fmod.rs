use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise floating-point remainder
pub fn fmod<
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
        .map(|(&i, &o)| i % o)
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

pub fn hypot<
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
        .map(|(&i, &o)| i.hypot(o))
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

pub fn ldexp<
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
        .map(|(&i, &o)| i * (T::from(2.0).unwrap().powf(o)))
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

pub fn remainder<
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
        .map(|(&i, &o)| {
            let r = i % o;
            if (r > T::zero() && o < T::zero()) || (r < T::zero() && o > T::zero()) {
                r + o
            } else {
                r
            }
        })
        .collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
