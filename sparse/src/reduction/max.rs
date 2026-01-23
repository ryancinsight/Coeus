//! Sparse max reduction
//!
//! Finds maximum non-zero element in sparse storage.

use dtype::DataType;
use storage::CsrStorage;

/// Trait for sparse max operation
pub trait SparseMax<T: DataType> {
    /// Find maximum non-zero element
    fn max_sparse(&self) -> Option<T>
    where
        T: PartialOrd + Copy;
}

impl<T: DataType> SparseMax<T> for CsrStorage<T> {
    fn max_sparse(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        self.data().iter().copied().reduce(|a, b| if a > b { a } else { b })
    }
}
