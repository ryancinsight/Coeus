//! Element-wise bitwise left shift operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use std::ops::Shl;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Element-wise bitwise left shift operation
pub fn bitwise_left_shift<
    T: DataType + Shl<Output = T> + Clone + 'static,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + StorageToDense<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape().dims() != b.shape().dims() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            actual: b.shape().dims().to_vec(),
            operation: "bitwise_left_shift",
        });
    }

    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| x.clone() << y.clone())
        .collect();

    Tensor::from_vec_with_backend(data, a.shape().dims(), a.backend.clone())
}
