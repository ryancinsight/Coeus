//! Sparse sigmoid activation
//!
//! Applies sigmoid activation to non-zero elements in sparse storage.
//! sigmoid(x) = 1 / (1 + exp(-x))


use dtype::DataType;
use storage::{CsrStorage, Result};

use backend::Backend;
use storage::{DenseStorage, Storage};
use alloc::format;

/// Trait for sparse sigmoid operation
pub trait SparseSigmoid<T: DataType> {
    /// Apply sigmoid to non-zero elements
    fn sigmoid_sparse<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

impl<T: DataType + Default> SparseSigmoid<T> for CsrStorage<T> {
    fn sigmoid_sparse<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Treat values as dense storage and apply backend operation
        let values_dense = DenseStorage::from_vec(self.data().to_vec(), &[self.data().len()])
            .map_err(|e| storage::StorageError::BackendError(format!("Dense conversions error: {}", e)))?;
            
        // We use backend.sigmoid_dense if it exists, or unary_op_gpu(6)
        // Check if Backend has sigmoid_dense. Let's look at Backend trait.
        let new_values_dense = backend.sigmoid_dense(&values_dense)
            .map_err(|e| storage::StorageError::BackendError(format!("Backend error: {}", e)))?;
        
        let new_data = new_values_dense.as_slice().to_vec();
        
        CsrStorage::new(
            new_data,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape_ref().dims(),
        )
    }
}
