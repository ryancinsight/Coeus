//! Sparse matrix element-wise operations
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use alloc::vec::Vec;
use storage::{CsrStorage, DataType, Storage};

/// Sparse matrix element-wise operations trait
pub trait SparseElementWise<T: DataType> {
    /// Apply a function to all non-zero elements
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(T) -> T;

    /// Absolute value of non-zero elements
    fn abs_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Signed;

    /// Sine of non-zero elements
    fn sin_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Cosine of non-zero elements
    fn cos_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Tangent of non-zero elements
    fn tan_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Hyperbolic tangent of non-zero elements
    fn tanh_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Ceiling of non-zero elements
    fn ceil_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Floor of non-zero elements
    fn floor_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Round non-zero elements
    fn round_sparse(&self) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

/// CSR element-wise operations implementation
impl<T: DataType + Copy> SparseElementWise<T> for CsrStorage<T> {
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        F: Fn(T) -> T,
    {
        let new_data: Vec<T> = self.data().iter().map(|&x| op(x)).collect();
        CsrStorage::new(
            new_data,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        )
    }

    fn abs_sparse(&self) -> Result<Self>
    where
        T: num_traits::Signed,
    {
        self.map_nz(|x| x.abs())
    }

    fn sin_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.sin())
    }

    fn cos_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Note: Cos(0) = 1, so this is NOT sparsity preserving.
        // However, map_nz only applies to non-zero elements.
        // If we want a mathematically correct cos(Sparse), it should be dense.
        // For now, we follow the "op on non-zeros" pattern if called via SparseElementWise,
        // but the dispatch layer should handle the dense conversion if correctness is required.
        self.map_nz(|x| x.cos())
    }

    fn tan_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.tan())
    }

    fn tanh_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.tanh())
    }

    fn ceil_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.ceil())
    }

    fn floor_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.floor())
    }

    fn round_sparse(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.map_nz(|x| x.round())
    }
}
