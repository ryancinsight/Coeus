//! Element-wise inverse tangent (y/x)

use crate::ops::arithmetic::broadcast_binary_op;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Element-wise inverse tangent (y/x)
pub fn atan2<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    y: &Tensor<B, S, T>,
    x: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    broadcast_binary_op(y, x, |y_val, x_val| y_val.atan2(x_val))
}
