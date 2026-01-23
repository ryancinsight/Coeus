//! Sigmoid activation for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// Sigmoid activation trait
pub trait DenseSigmoid<T: DataType> {
    /// Apply sigmoid activation: 1 / (1 + exp(-x))
    fn sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError>;
}

impl<T: DataType + num_traits::Float> DenseSigmoid<T> for DenseStorage<T> {
    fn sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError> {
        backend.sigmoid_dense(self)
    }
}
