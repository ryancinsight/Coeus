use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

use super::Activation;

/// Hardsigmoid activation: clamp((x + 3) / 6, 0, 1)
#[derive(Debug, Clone)]
pub struct Hardsigmoid<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for Hardsigmoid<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Hardsigmoid<B, S, T>
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

impl<B, S, T> Activation<B, S, T> for Hardsigmoid<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let three = Tensor::<B, S, T>::full_like(x, T::from_f64(3.0).unwrap())?;
        let six = Tensor::<B, S, T>::full_like(x, T::from_f64(6.0).unwrap())?;
        let zero = Tensor::<B, S, T>::zeros_like(x)?;
        let one = Tensor::<B, S, T>::ones_like(x)?;

        // (x + 3) / 6
        let x_plus_3 = add(x, &three)?;
        let div_6 = div(&x_plus_3, &six)?;

        // clamp(..., 0, 1) -> min(max(..., 0), 1)
        let clamped_min = maximum(&div_6, &zero)?;
        let res = minimum(&clamped_min, &one)?;

        Ok(res)
    }
}

impl<B, S, T> Module<B, S, T> for Hardsigmoid<B, S, T>
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
        "Hardsigmoid"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
