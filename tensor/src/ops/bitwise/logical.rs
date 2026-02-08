//! Logical operations
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::Storage;

pub fn logical_and<
    T: DataType + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + 'static,
>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = lhs.storage.storage_logical_and(&rhs.storage, &lhs.backend)?;
    Ok(Tensor::from_storage(storage, lhs.backend.clone()))
}

pub fn logical_or<
    T: DataType + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + 'static,
>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = lhs.storage.storage_logical_or(&rhs.storage, &lhs.backend)?;
    Ok(Tensor::from_storage(storage, lhs.backend.clone()))
}

pub fn logical_xor<
    T: DataType + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + 'static,
>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = lhs.storage.storage_logical_xor(&rhs.storage, &lhs.backend)?;
    Ok(Tensor::from_storage(storage, lhs.backend.clone()))
}

pub fn logical_not<
    T: DataType + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + 'static,
>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = input.storage.storage_logical_not(&input.backend)?;
    Ok(Tensor::from_storage(storage, input.backend.clone()))
}
