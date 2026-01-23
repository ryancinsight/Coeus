//! Sparse GELU activation

use dtype::DataType;
use storage::{CsrStorage, Result};
use backend::Backend;
use crate::arithmetic::SparseElementWise;

/// Trait for sparse GELU operation
pub trait SparseGelu<T: DataType> {
    /// Apply GELU to non-zero elements
    fn gelu_sparse<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

impl<T: DataType + Default + Copy + num_traits::Float> SparseGelu<T> for CsrStorage<T> {
    fn gelu_sparse<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        let half = T::from(0.5).unwrap();
        let one = T::from(1.0).unwrap();
        let sqrt_2_over_pi = T::from((2.0 / core::f64::consts::PI).sqrt()).unwrap();
        let coeff = T::from(0.044715).unwrap();

        self.map_nz(|x| {
            let x_cubed = x * x * x;
            let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            let tanh_inner = inner.tanh();
            half * x * (one + tanh_inner)
        })
    }
}
