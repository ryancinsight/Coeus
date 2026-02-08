//! Element-wise comparison operations
//!
//! This module provides element-wise comparison operations for tensors:
//! - eq, ne: Equal, Not Equal
//! - gt, ge: Greater Than, Greater Expected
//! - lt, le: Less Than, Less Expected
//!
//! These operations return a tensor of the same shape with binary 0/1 values
//! (represented as the input data type T, typically 0.0/1.0 for floats).

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

pub mod where_cond;
pub mod isclose;
pub mod allclose;
pub mod isnan;
pub mod isinf;
pub mod isfinite;

pub use isclose::isclose;
pub use allclose::allclose;
pub use where_cond::where_cond;
pub use isnan::isnan;
pub use isinf::isinf;
pub use isfinite::isfinite;

pub mod scalar;
pub use scalar::*;

pub mod maximum_scalar;
pub mod minimum_scalar;
pub use maximum_scalar::maximum_scalar;
pub use minimum_scalar::minimum_scalar;


use crate::ops::TensorStorageOps;

/// Element-wise equality comparison
pub fn eq<
    T: DataType + PartialEq + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_eq(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}

/// Element-wise inequality comparison
pub fn ne<
    T: DataType + PartialEq + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_ne(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}

/// Element-wise greater than comparison
pub fn gt<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_gt(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}

/// Element-wise greater than or equal comparison
pub fn ge<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_ge(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}

/// Element-wise less than comparison
pub fn lt<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_lt(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}

/// Element-wise less than or equal comparison
pub fn le<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let storage = a.storage.storage_le(&b.storage, &a.backend)?;
    Ok(Tensor::from_storage(storage, a.backend.clone()))
}
