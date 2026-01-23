//! Tanh activation for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// Tanh activation trait
pub trait DenseTanh<T: DataType> {
    /// Apply tanh activation
    fn tanh_activation<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError>;
}

impl<T: DataType + num_traits::Float> DenseTanh<T> for DenseStorage<T> {
    fn tanh_activation<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError> {
        backend.tanh_dense(self)
    }
}
