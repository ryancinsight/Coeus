//! Sparse activation functions
//!
//! This module provides activation functions for sparse storage types.
//! Activations are applied only to non-zero elements for efficiency.

pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod gelu;

pub use relu::SparseRelu;
pub use sigmoid::SparseSigmoid;
pub use tanh::SparseTanh;
pub use gelu::SparseGelu;

use backend::Backend;
use storage::{CsrStorage, Result};
use dtype::DataType;

/// Unified trait for sparse activation operations
pub trait SparseActivation<T: DataType> {
    /// Apply ReLU activation to non-zero elements
    fn activation_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: PartialOrd + Default + Copy;

    /// Apply sigmoid activation to non-zero elements
    fn activation_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Apply tanh activation to non-zero elements
    fn activation_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;

    /// Apply GELU activation to non-zero elements
    fn activation_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Float;
}

impl<T: DataType + Default> SparseActivation<T> for CsrStorage<T> {
    fn activation_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default + Copy,
    {
        SparseRelu::relu_sparse(self, backend)
    }

    fn activation_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        SparseSigmoid::sigmoid_sparse(self, backend)
    }

    fn activation_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        SparseTanh::tanh_sparse(self, backend)
    }

    fn activation_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        SparseGelu::gelu_sparse(self, backend)
    }
}
