//! Sparse mean reduction
//!
//! Computes mean of non-zero elements in sparse storage.

use dtype::DataType;
use storage::CsrStorage;

/// Trait for sparse mean operation
pub trait SparseMean<T: DataType> {
    /// Compute mean of non-zero elements
    fn mean_sparse(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + num_traits::FromPrimitive;
}

impl<T: DataType> SparseMean<T> for CsrStorage<T> {
    fn mean_sparse(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + num_traits::FromPrimitive,
    {
        let nnz = self.nnz();
        if nnz == 0 {
            return T::default();
        }
        let sum: T = self.data().iter().copied().fold(T::default(), |acc, x| acc + x);
        let count = T::from_usize(nnz).unwrap_or(T::from_usize(1).unwrap());
        sum / count
    }
}
