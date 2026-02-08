//! Logical any reduction operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Tests if any element along specified dimensions evaluates to true.
///
/// For numeric tensors, any non-zero value is considered true.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dims` - Dimensions to reduce over. None means all dimensions.
/// * `keepdim` - Whether to keep the reduced dimensions
pub fn any<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + num_traits::One + num_traits::Zero + PartialOrd + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static + crate::ops::dispatch::TensorStorageOps<T>,
{
    let init = T::zero();
    let op = |acc: T, val: T| {
        if acc != T::zero() || val != T::zero() {
            T::one()
        } else {
            T::zero()
        }
    };
    
    super::reduce_dims(tensor, dims, keepdim, op, init)
}
