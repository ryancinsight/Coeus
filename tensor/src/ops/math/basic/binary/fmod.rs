//! Element-wise modulo and remainder operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise remainder of division (C++ fmod)
pub fn fmod<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&a, &b)| a % b).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

/// Element-wise remainder of division (Python %)
pub fn remainder<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&a, &b)| {
        let r = a % b;
        if (r > T::zero() && b < T::zero()) || (r < T::zero() && b > T::zero()) { r + b } else { r }
    }).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

/// Element-wise hypot (sqrt(x^2 + y^2))
pub fn hypot<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&a, &b)| a.hypot(b)).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}

/// Element-wise ldexp (x * 2^exp)
pub fn ldexp<T, B, S>(input: &Tensor<B, S, T>, other: &Tensor<B, S, i32>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = input.as_slice().iter().zip(other.as_slice().iter()).map(|(&a, &b)| a * T::from_i32(2).unwrap().powi(*b)).collect();
    Tensor::from_vec_with_backend(data, input.shape().dims(), input.backend.clone())
}
