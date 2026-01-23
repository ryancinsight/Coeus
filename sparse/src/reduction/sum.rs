//! Sparse sum reduction
//!
//! Sums all non-zero elements in sparse storage.

use dtype::DataType;
use storage::CsrStorage;

/// Trait for sparse sum operation
pub trait SparseSum<T: DataType> {
    /// Sum all non-zero elements
    fn sum_sparse(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + Copy;
}

impl<T: DataType> SparseSum<T> for CsrStorage<T> {
    fn sum_sparse(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + Copy,
    {
        self.data().iter().copied().fold(T::default(), |acc, x| acc + x)
    }
}
