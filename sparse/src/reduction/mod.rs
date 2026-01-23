//! Sparse reduction operations
//!
//! This module provides reduction operations for sparse storage types.
//! Reductions operate on non-zero values only for efficiency.

pub mod sum;
pub mod max;
pub mod min;
pub mod mean;

pub use sum::SparseSum;
pub use max::SparseMax;
pub use min::SparseMin;
pub use mean::SparseMean;

use storage::CsrStorage;
use dtype::DataType;

/// Unified trait for sparse reduction operations
pub trait SparseReduce<T: DataType> {
    /// Sum of all non-zero elements
    fn reduce_sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + Copy;

    /// Maximum non-zero element
    fn reduce_max(&self) -> Option<T>
    where
        T: PartialOrd + Copy;

    /// Minimum non-zero element
    fn reduce_min(&self) -> Option<T>
    where
        T: PartialOrd + Copy;

    /// Mean of non-zero elements
    fn reduce_mean(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + num_traits::FromPrimitive;
}

impl<T: DataType> SparseReduce<T> for CsrStorage<T> {
    fn reduce_sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + Copy,
    {
        SparseSum::sum_sparse(self)
    }

    fn reduce_max(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        SparseMax::max_sparse(self)
    }

    fn reduce_min(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        SparseMin::min_sparse(self)
    }

    fn reduce_mean(&self) -> T
    where
        T: Default + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + num_traits::FromPrimitive,
    {
        SparseMean::mean_sparse(self)
    }
}
