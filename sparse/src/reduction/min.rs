//! Sparse min reduction
//!
//! Finds minimum non-zero element in sparse storage.

use dtype::DataType;
use storage::CsrStorage;

/// Trait for sparse min operation
pub trait SparseMin<T: DataType> {
    /// Find minimum non-zero element
    fn min_sparse(&self) -> Option<T>
    where
        T: PartialOrd + Copy;
}

impl<T: DataType> SparseMin<T> for CsrStorage<T> {
    fn min_sparse(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        self.data().iter().copied().reduce(|a, b| if a < b { a } else { b })
    }
}
