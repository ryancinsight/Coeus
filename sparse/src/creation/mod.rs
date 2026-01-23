//! Creation operations for sparse storage
//!
//! All sparse operations use CSR format as the single optimized format.

pub mod zeros;

pub use zeros::{zeros, zeros_csr, zeros_csc};

use dtype::DataType;
use storage::{CsrStorage, Result};

/// Trait for sparse tensor creation operations
pub trait SparseCreation<T: DataType> {
    /// Create a sparse tensor filled with zeros (conceptually)
    ///
    /// For sparse tensors, this creates an empty storage with the given shape.
    fn zeros(shape: &[usize]) -> Result<Self>
    where
        Self: Sized;
}

impl<T: DataType + Copy> SparseCreation<T> for CsrStorage<T> {
    fn zeros(shape: &[usize]) -> Result<Self> {
        zeros::zeros(shape)
    }
}
