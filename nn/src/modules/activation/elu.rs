use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

use super::Activation;

/// ELU activation: x > 0 ? x : alpha * (exp(x) - 1)
#[derive(Debug, Clone)]
pub struct ELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    alpha: T,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for ELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new(T::from_f64(1.0).unwrap())
    }
}

impl<B, S, T> ELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    pub fn new(alpha: T) -> Self {
        Self {
            alpha,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Activation<B, S, T> for ELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // ops::activation::elu returns DenseStorage, so we convert back
        let y = crate::ops::activation::elu(x, self.alpha)?;
        let storage = S::from_vec(y.as_slice().to_vec(), y.shape().dims())?;
        Ok(Tensor::from_storage(storage, x.backend().clone()))
    }
}

impl<B, S, T> Module<B, S, T> for ELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        <Self as Activation<B, S, T>>::forward(self, input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "ELU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
