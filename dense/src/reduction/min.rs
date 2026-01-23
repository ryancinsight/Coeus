//! Min reduction for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// Min reduction trait
pub trait DenseMin<T: DataType> {
    /// Compute minimum element
    fn min<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError>;
}

impl<T: DataType + PartialOrd + Copy> DenseMin<T> for DenseStorage<T> {
    fn min<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError> {
        backend.min_dense(self)
    }
}
