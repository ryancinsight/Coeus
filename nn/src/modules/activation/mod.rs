//! Advanced Activation Functions for Neural Networks
//!
//! Implementation of state-of-the-art activation functions including:
//! - SwiGLU (Swish-Gated Linear Unit)
//! - GeLU variants
//! - SiLU/Swish
//! - ReLU/PReLU

pub mod elu;
pub mod gelu;
pub mod hard_tanh;
pub mod hardshrink;
pub mod hardsigmoid;
pub mod hardswish;
pub mod leaky_relu;
pub mod logsigmoid;
pub mod mish;
pub mod prelu;
pub mod relu;
pub mod relu6;
pub mod selu;
pub mod silu;
pub mod soft_plus;
pub mod softshrink;
pub mod swiglu;

pub use elu::ELU;
pub use gelu::GeLU;
pub use hard_tanh::Hardtanh;
pub use hardshrink::Hardshrink;
pub use hardsigmoid::Hardsigmoid;
pub use hardswish::Hardswish;
pub use leaky_relu::LeakyReLU;
pub use logsigmoid::LogSigmoid;
pub use mish::Mish;
pub use prelu::PReLU;
pub use relu::ReLU;
pub use relu6::ReLU6;
pub use selu::SELU;
pub use silu::SiLU;
pub use soft_plus::Softplus;
pub use softshrink::Softshrink;
pub use swiglu::SwiGLU;

use crate::core::error::Result;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

/// Activation function registry for dynamic activation selection
pub enum ActivationType<T> {
    SwiGLU,
    GeLU,
    SiLU,
    ReLU,
    PReLU(usize, Option<T>),
    LeakyReLU(T),
    ELU(T),
    Hardtanh(T, T),
    // Softplus(T, T),
    Mish,
}

pub struct ActivationFactory<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> ActivationFactory<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + num_traits::Num
        + Copy
        + num_traits::FromPrimitive,
{
    /// Create an activation function by type
    pub fn create(activation_type: ActivationType<T>) -> Box<dyn Activation<B, S, T>> {
        match activation_type {
            ActivationType::SwiGLU => Box::new(SwiGLU::new()),
            ActivationType::GeLU => Box::new(GeLU::new()),
            ActivationType::SiLU => Box::new(SiLU::new()),
            ActivationType::ReLU => Box::new(ReLU::new()),
            ActivationType::PReLU(num_params, init) => Box::new(PReLU::new(num_params, init)),
            ActivationType::LeakyReLU(slope) => Box::new(LeakyReLU::new(slope)),
            ActivationType::ELU(alpha) => Box::new(ELU::new(alpha)),
            ActivationType::Hardtanh(min, max) => Box::new(Hardtanh::new(min, max)),
            // ActivationType::Softplus(beta, threshold) => Box::new(Softplus::new(beta.clone(), threshold.clone())),
            ActivationType::Mish => Box::new(Mish::new()),
        }
    }
}

/// Common activation trait for polymorphism
pub trait Activation<B, S, T>: Send + Sync
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestDataType = Float32;

    #[test]
    fn test_swiglu_basic() {
        let _swiglu = SwiGLU::<TestBackend, TestStorage, TestDataType>::new();
        // ... test content ...
    }

    /*
    #[test]
    fn test_activation_factory() {
        let relu = ActivationFactory::create(ActivationType::ReLU);
        // ...
    }
    */
}
