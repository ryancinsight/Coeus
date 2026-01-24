use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

use super::Activation;

/// ReLU6 activation: min(max(0, x), 6)
#[derive(Debug, Clone)]
pub struct ReLU6<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for ReLU6<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> ReLU6<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Activation<B, S, T> for ReLU6<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let zero = Tensor::<B, S, T>::zeros_like(x)?;
        let six = Tensor::<B, S, T>::full_like(x, T::from_f64(6.0).unwrap())?;
        let val = maximum(x, &zero)?;
        Ok(minimum(&val, &six)?)
    }
}

impl<B, S, T> Module<B, S, T> for ReLU6<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        <Self as Activation<B, S, T>>::forward(self, input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "ReLU6"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
