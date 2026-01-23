//! GELU activation for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::DenseStorage;

/// GELU activation trait
pub trait DenseGelu<T: DataType> {
    /// Apply GELU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError>;
}

impl<T: DataType + num_traits::Float + num_traits::FromPrimitive> DenseGelu<T> for DenseStorage<T> {
    fn gelu<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<DenseStorage<T>, BackendError> {
        backend.gelu_dense(self)
    }
}
