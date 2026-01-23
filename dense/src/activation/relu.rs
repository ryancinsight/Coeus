//! ReLU activation for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// ReLU activation trait
pub trait DenseRelu<T: DataType> {
    /// Apply ReLU activation: max(0, x)
    fn relu<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError>;
}

impl<T: DataType + num_traits::Zero + PartialOrd + Copy> DenseRelu<T> for DenseStorage<T> {
    fn relu<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError> {
        backend.relu_dense(self)
    }
}
