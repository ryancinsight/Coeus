//! Sparse tensor reduction operations
//!
//! All sparse operations use CSR format as the single optimized format.

use storage::{CsrStorage, DataType, Storage};

/// Sparse tensor reduction operations trait
pub trait SparseReduce<T: DataType>
where
    T: num_traits::Zero + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd,
{
    /// Sum of all elements (including implicit zeros)
    fn sum(&self) -> T;
    /// Mean of all elements
    fn mean(&self) -> T;
    /// Maximum element value
    fn max(&self) -> T;
    /// Minimum element value
    fn min(&self) -> T;
    /// Sum of non-zero elements only
    fn sum_nz(&self) -> T;
    /// Number of non-zero elements
    fn nnz(&self) -> usize;
    /// Sparsity ratio (1 - nnz/total)
    fn sparsity(&self) -> f64;
}

/// CSR reduction operations implementation
impl<T> SparseReduce<T> for CsrStorage<T>
where
    T: DataType + num_traits::Zero + num_traits::One + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd,
{
    fn sum(&self) -> T {
        self.sum_nz()
    }

    #[allow(clippy::cast_precision_loss)]
    fn mean(&self) -> T {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            T::zero()
        } else {
            let total_sum = self.sum();
            total_sum / T::from_usize(total_elements).unwrap_or(T::one())
        }
    }

    fn max(&self) -> T {
        self.data()
            .iter()
            .fold(T::zero(), |acc, &x| if x > acc { x } else { acc })
    }

    fn min(&self) -> T {
        self.data()
            .iter()
            .fold(T::zero(), |acc, &x| if x < acc { x } else { acc })
    }

    fn sum_nz(&self) -> T {
        self.data().iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn nnz(&self) -> usize {
        self.data().len()
    }

    #[allow(clippy::cast_precision_loss)]
    fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }
}
