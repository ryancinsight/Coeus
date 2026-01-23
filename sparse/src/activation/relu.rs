//! Sparse ReLU activation
//!
//! Applies ReLU activation to non-zero elements in sparse storage.
//! ReLU(x) = max(0, x)

use dtype::DataType;
use storage::{CsrStorage, Result};
use backend::Backend;
use crate::arithmetic::SparseElementWise;

/// Trait for sparse ReLU operation
pub trait SparseRelu<T: DataType> {
    /// Apply ReLU to non-zero elements
    fn relu_sparse<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: PartialOrd + Default + Copy;
}

impl<T: DataType + Default + Copy + PartialOrd> SparseRelu<T> for CsrStorage<T> {
    fn relu_sparse<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        self.map_nz(|val| {
            if val > T::default() {
                val
            } else {
                T::default()
            }
        })
    }
}
