//! Dense reduction operations
//!
//! Sum, max, min, mean reductions for dense tensors.

mod sum;
mod max;
mod min;
mod mean;

pub use sum::DenseSum;
pub use max::DenseMax;
pub use min::DenseMin;
pub use mean::DenseMean;

use dtype::DataType;

/// Unified reduction trait for dense tensors
pub trait DenseReduce<T: DataType>: DenseSum<T> + DenseMax<T> + DenseMin<T> + DenseMean<T> {}

impl<T, S> DenseReduce<T> for S
where
    T: DataType,
    S: DenseSum<T> + DenseMax<T> + DenseMin<T> + DenseMean<T>,
{}
