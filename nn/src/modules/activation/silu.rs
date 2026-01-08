use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

use super::swiglu::SwiGLU;
use super::{Activation, ActivationType};

/// SiLU (Sigmoid Linear Unit) activation function
///
/// SiLU(x) = x * sigmoid(x)
/// Also known as Swish: Swish(x) = x * sigmoid(x)
#[derive(Debug, Clone)]
pub struct SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    swiglu: SwiGLU<B, S, T>,
}

impl<B, S, T> Default for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new SiLU activation function
    pub fn new() -> Self {
        Self {
            swiglu: SwiGLU::new(),
        }
    }

    /// Apply SiLU activation: x * sigmoid(x)
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // SiLU(x) = x * sigmoid(x), which is equivalent to SwiGLU(x, x)
        self.swiglu.forward(x, x)
    }
}

impl<B, S, T> Activation<B, S, T> for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        SiLU::<B, S, T>::forward(self, x)
    }
}

impl<B, S, T> Module<B, S, T> for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "SiLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
