//! Sparse tanh activation
//!
//! Applies tanh activation to non-zero elements in sparse storage.
//! tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

use dtype::DataType;
use storage::{CsrStorage, Result};
use backend::Backend;
use crate::arithmetic::SparseElementWise;

/// Trait for sparse tanh operation
pub trait SparseTanh<T: DataType> {
    /// Apply tanh to non-zero elements
    fn tanh_sparse<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

impl<T: DataType + Default + Copy + num_traits::Float> SparseTanh<T> for CsrStorage<T> {
    fn tanh_sparse<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        self.map_nz(|val| val.tanh())
    }
}
