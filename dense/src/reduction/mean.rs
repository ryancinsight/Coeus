//! Mean reduction for dense tensors

use backend::{Backend, BackendError};
use dtype::DataType;
use storage::{DenseStorage, Storage};

/// Mean reduction trait
pub trait DenseMean<T: DataType> {
    /// Compute mean of all elements
    fn mean<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError>;
}

impl<T: DataType + num_traits::Zero + num_traits::One + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy> 
    DenseMean<T> for DenseStorage<T> 
{
    fn mean<B: Backend<Data = T>>(&self, backend: &B) -> core::result::Result<T, BackendError> {
        let sum = backend.sum_dense(self)?;
        let count = T::from_usize(self.as_slice().len()).unwrap_or(T::one());
        Ok(sum / count)
    }
}
