//! Dense activation operations
//!
//! Activation functions for dense tensors: relu, sigmoid, tanh, gelu.

mod relu;
mod sigmoid;
mod tanh;
mod gelu;

pub use relu::DenseRelu;
pub use sigmoid::DenseSigmoid;
pub use tanh::DenseTanh;
pub use gelu::DenseGelu;

use dtype::DataType;

/// Unified activation trait for dense tensors
pub trait DenseActivation<T: DataType>: 
    DenseRelu<T> + DenseSigmoid<T> + DenseTanh<T> + DenseGelu<T> 
{}

impl<T, S> DenseActivation<T> for S
where
    T: DataType,
    S: DenseRelu<T> + DenseSigmoid<T> + DenseTanh<T> + DenseGelu<T>,
{}
