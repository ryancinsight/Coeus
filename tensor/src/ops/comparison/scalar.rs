//! Scalar comparison operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise equality comparison with scalar
pub fn eq_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialEq + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x == scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise inequality comparison with scalar
pub fn ne_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialEq + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x != scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise greater than comparison with scalar
pub fn gt_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x > scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise greater than or equal comparison with scalar
pub fn ge_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x >= scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise less than comparison with scalar
pub fn lt_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x < scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise less than or equal comparison with scalar
pub fn le_scalar<B, S, T>(tensor: &Tensor<B, S, T>, scalar: T) -> Result<Tensor<B, S, T>>
where
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x <= scalar { T::one() } else { T::zero() })
        .collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
