//! Maximum operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise maximum with broadcasting.
pub fn maximum<
    T: DataType + PartialOrd + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    super::broadcast_binary_op(a, b, |x, y| if x > y { x } else { y })
}
