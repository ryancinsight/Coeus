//! Element-wise sign operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Signed;
use storage::{Storage, StorageFromVec};

/// Element-wise sign
pub fn sign<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Signed + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<T> = tensor.as_slice().iter().map(|&x| {
        if x.is_zero() { T::zero() }
        else if x.is_positive() { T::one() }
        else { -T::one() }
    }).collect();
    
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}

/// Element-wise signbit
pub fn signbit<T, B, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, bool>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Signed + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let data: Vec<bool> = tensor.as_slice().iter().map(|&x| x.is_negative()).collect();
    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
