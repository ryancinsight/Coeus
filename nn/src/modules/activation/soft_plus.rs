use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::{self, arithmetic::*}, FloatExt, Tensor};

use super::Activation;

/// Softplus activation: (1/beta) * log(1 + exp(beta * x))
#[derive(Debug, Clone)]
pub struct Softplus<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    beta: T,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Default for Softplus<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn default() -> Self {
        let beta = T::from_f64(1.0).unwrap();
        let threshold = T::from_f64(20.0).unwrap();
        Self::new(beta, threshold)
    }
}

impl<B, S, T> Softplus<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    pub fn new(beta: T, _threshold: T) -> Self {
        Self {
            beta,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Activation<B, S, T> for Softplus<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let beta_t = Tensor::<B, S, T>::full_like(x, self.beta)?;
        // Threshold check omitted for stability unless required

        let bx = mul(x, &beta_t)?;

        let exp_bx = ops::exp(&bx)?;
        let one = Tensor::<B, S, T>::ones_like(x)?;
        let one_plus_exp = add(&one, &exp_bx)?;
        let log_res = ops::log(&one_plus_exp)?;

        let one_val = T::from_f64(1.0).unwrap();
        let inv_beta_val = one_val / self.beta;
        let inv_beta = Tensor::<B, S, T>::full_like(x, inv_beta_val)?;
        Ok(mul(&inv_beta, &log_res)?)
    }
}

impl<B, S, T> Module<B, S, T> for Softplus<B, S, T>
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
        "Softplus"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
