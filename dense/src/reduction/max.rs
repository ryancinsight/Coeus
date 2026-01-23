//! Max reduction for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// Max reduction trait
pub trait DenseMax<T: DataType> {
    /// Compute maximum element
    fn max<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError>;
}

impl<T: DataType + PartialOrd + Copy> DenseMax<T> for DenseStorage<T> {
    fn max<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError> {
        backend.max_dense(self)
    }
}
