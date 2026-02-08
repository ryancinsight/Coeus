use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

use super::Activation;

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
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new SiLU activation function
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply SiLU activation: x * sigmoid(x)
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let y = crate::ops::activation::silu(x)?;
        // Wrap output in tensor with correct storage if needed, or if silu returns correct type.
        // crate::ops::activation::silu returns Tensor<B, DenseStorage<T>, T> usually if ported from functional.
        // Wait, my implementation of silu in ops/activation.rs returns Tensor<B, DenseStorage<T>, T> because it does to_dense.
        // But SiLU module expects Result<Tensor<B, S, T>>.
        // I need to convert back to storage S.
        let storage = S::from_vec(y.as_slice().to_vec(), y.shape().dims())?;
        Ok(Tensor::from_storage(storage, x.backend().clone()))
    }
}

impl<B, S, T> Activation<B, S, T> for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        SiLU::<B, S, T>::forward(self, x)
    }
}

impl<B, S, T> Module<B, S, T> for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "SiLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
