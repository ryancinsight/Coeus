//! Sparse layout operations
//!
//! All sparse operations use CSR format as the single optimized format.

pub mod reshape;
pub mod transpose;

pub use reshape::SparseReshape;
pub use transpose::SparseTranspose;

use dtype::DataType;
use storage::{CsrStorage, Result};

/// Trait for sparse storage layout operations
pub trait SparseLayout<T: DataType>: SparseTranspose<T> + SparseReshape<T> {
    /// Transpose the sparse matrix
    fn transpose(&self) -> Result<CsrStorage<T>>
    where
        Self: Sized;

    /// Reshape the sparse matrix
    fn reshape(&self, new_shape: &[usize]) -> Result<CsrStorage<T>>
    where
        Self: Sized;
}

impl<T: DataType + Default + Copy> SparseLayout<T> for CsrStorage<T> {
    fn transpose(&self) -> Result<CsrStorage<T>> {
        SparseTranspose::transpose_sparse(self)
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<CsrStorage<T>> {
        SparseReshape::reshape_sparse(self, new_shape)
    }
}
