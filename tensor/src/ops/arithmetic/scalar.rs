//! Scalar arithmetic operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Scalar addition
pub fn add_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + std::ops::Add<Output = T> + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x + scalar).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Scalar addition in-place
pub fn add_scalar_<S, T>(storage: &mut S, scalar: T) -> Result<()>
where
    T: DataType + std::ops::Add<Output = T> + Copy,
    S: Storage<T>,
{
    let data = storage.as_mut_slice();
    for x in data {
        *x = *x + scalar;
    }
    Ok(())
}

/// Scalar subtraction
pub fn sub_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + std::ops::Sub<Output = T> + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x - scalar).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Scalar subtraction in-place
pub fn sub_scalar_<S, T>(storage: &mut S, scalar: T) -> Result<()>
where
    T: DataType + std::ops::Sub<Output = T> + Copy,
    S: Storage<T>,
{
    let data = storage.as_mut_slice();
    for x in data {
        *x = *x - scalar;
    }
    Ok(())
}

/// Scalar multiplication
pub fn mul_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + std::ops::Mul<Output = T> + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x * scalar).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Scalar multiplication in-place
pub fn mul_scalar_<S, T>(storage: &mut S, scalar: T) -> Result<()>
where
    T: DataType + std::ops::Mul<Output = T> + Copy,
    S: Storage<T>,
{
    let data = storage.as_mut_slice();
    for x in data {
        *x = *x * scalar;
    }
    Ok(())
}

/// Scalar division
pub fn div_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + std::ops::Div<Output = T> + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| x / scalar).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Scalar division in-place
pub fn div_scalar_<S, T>(storage: &mut S, scalar: T) -> Result<()>
where
    T: DataType + std::ops::Div<Output = T> + Copy,
    S: Storage<T>,
{
    let data = storage.as_mut_slice();
    for x in data {
        *x = *x / scalar;
    }
    Ok(())
}
