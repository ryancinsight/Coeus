//! Sum reduction for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// Sum reduction trait
pub trait DenseSum<T: DataType> {
    /// Compute sum of all elements
    fn sum<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError>;
}

impl<T: DataType + num_traits::Zero + core::ops::Add<Output = T> + Copy> DenseSum<T> for DenseStorage<T> {
    fn sum<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError> {
        backend.sum_dense(self)
    }
}
