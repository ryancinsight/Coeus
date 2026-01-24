use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::{self, arithmetic::*}, FloatExt, Tensor};

use super::Activation;

/// SELU activation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
#[derive(Debug, Clone)]
pub struct SELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for SELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> SELU<B, S, T>
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

impl<B, S, T> Activation<B, S, T> for SELU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        const SCALE: f64 = 1.050_700_987_355_480_5;

        let zero = Tensor::<B, S, T>::zeros_like(x)?;
        let alpha_t = Tensor::<B, S, T>::full_like(x, T::from_f64(ALPHA).unwrap())?;
        let scale_t = Tensor::<B, S, T>::full_like(x, T::from_f64(SCALE).unwrap())?;

        // Positive part: max(0, x)
        let pos = maximum(x, &zero)?;

        // Negative part: min(0, alpha * (exp(x) - 1))
        let exp_x = ops::exp(x)?;
        let one = Tensor::<B, S, T>::ones_like(x)?;
        let exp_minus_1 = sub(&exp_x, &one)?;
        let scaled_exp = mul(&alpha_t, &exp_minus_1)?;
        let neg = minimum(&zero, &scaled_exp)?;

        let sum = add(&pos, &neg)?;
        Ok(mul(&scale_t, &sum)?)
    }
}

impl<B, S, T> Module<B, S, T> for SELU<B, S, T>
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
        "SELU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
