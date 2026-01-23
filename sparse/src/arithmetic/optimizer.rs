//! Sparse optimizer operations
//!
//! Specialized sparse operations for gradient-based optimizers.
//! All operations use CSR format.

use crate::Result;
use alloc::vec::Vec;
use storage::{CsrStorage, DataType, Storage};

/// Sparse optimizer operations trait
pub trait SparseOptimizerOps<T: DataType> {
    /// Compute gradient norm (L2 norm of sparse vector)
    fn gradient_norm(&self) -> T
    where
        T: num_traits::Float;

    /// Clip gradient values by magnitude
    fn clip_by_value(&self, min_val: T, max_val: T) -> Result<Self>
    where
        Self: Sized,
        T: PartialOrd + Copy;

    /// Apply element-wise square root to gradients
    fn sqrt_nz(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Apply element-wise sqrt with epsilon for numerical stability
    fn sqrt_eps(&self, eps: T) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

/// CSR optimizer operations implementation
impl<T: DataType + Copy> SparseOptimizerOps<T> for CsrStorage<T> {
    fn gradient_norm(&self) -> T
    where
        T: num_traits::Float,
    {
        let sum_sq = self
            .data()
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x);
        sum_sq.sqrt()
    }

    fn clip_by_value(&self, min_val: T, max_val: T) -> Result<Self>
    where
        T: PartialOrd + Copy,
    {
        let clipped: Vec<T> = self
            .data()
            .iter()
            .map(|&x| {
                if x < min_val {
                    min_val
                } else if x > max_val {
                    max_val
                } else {
                    x
                }
            })
            .collect();

        CsrStorage::new(
            clipped,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        )
    }

    fn sqrt_nz(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let sqrts: Vec<T> = self.data().iter().map(|&x| x.sqrt()).collect();
        CsrStorage::new(
            sqrts,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        )
    }

    fn sqrt_eps(&self, eps: T) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let sqrts: Vec<T> = self.data().iter().map(|&x| (x + eps).sqrt()).collect();
        CsrStorage::new(
            sqrts,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        )
    }
}
