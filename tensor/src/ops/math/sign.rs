//! Sign and signbit operations
//!
//! Element-wise sign determination operations.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Returns a tensor with the signs of the elements of input.
///
/// The sign is:
/// - 1 if the element is greater than 0
/// - 0 if the element is 0
/// - -1 if the element is less than 0
pub fn sign<B, T, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let data = tensor.as_slice();
    let result: alloc::vec::Vec<T> = data
        .iter()
        .map(|&x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        })
        .collect();

    Tensor::from_vec_with_backend(result, tensor.shape().dims(), tensor.backend.clone())
}

/// Tests if each element of input has its sign bit set or not.
///
/// Returns true for negative numbers and -0.0, false otherwise.
pub fn signbit<B, T, S>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let data = tensor.as_slice();
    let result: alloc::vec::Vec<T> = data
        .iter()
        .map(|&x| {
            if x.is_sign_negative() {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();

    Tensor::from_vec_with_backend(result, tensor.shape().dims(), tensor.backend.clone())
}
