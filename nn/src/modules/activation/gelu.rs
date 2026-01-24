use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};
// Note: adjusting import path for functional
// use crate::ops::activation::gelu; // Removed unused

use super::Activation;

/// GeLU (Gaussian Error Linear Unit) activation function
///
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[derive(Debug, Clone)]
pub struct GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    /// Create a new GeLU activation function
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply GeLU activation using the tanh approximation
    ///
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Compute x^3
        let x_squared = mul(x, x)?;
        let x_cubed = mul(&x_squared, x)?;

        // Compute 0.044715 * x^3
        let coeff = T::from(0.044715).unwrap();
        let scaled_cubed = x_cubed.mul_scalar(coeff)?;

        // Compute x + 0.044715 * x^3
        let inner_term = add(x, &scaled_cubed)?;

        // Compute sqrt(2/π) ≈ 0.7978845608
        let sqrt_2_pi = T::from(0.7978845608).unwrap();
        let scaled_inner = inner_term.mul_scalar(sqrt_2_pi)?;

        // Compute tanh(scaled_inner)
        let scaled_inner_dense = scaled_inner.to_dense_generic()?;
        // Assuming functional_tanh is available
        let tanh_result_dense = crate::ops::activation::tanh(&scaled_inner_dense)?;
        let tanh_result_data = tanh_result_dense.as_slice().to_vec();
        let tanh_result =
            Tensor::<B, S, T>::from_vec(tanh_result_data, scaled_inner.shape().dims())?;

        // Compute 1 + tanh_result
        let one_data = vec![T::from(1.0).unwrap(); x.shape().dims().iter().product()];
        let one = Tensor::<B, S, T>::from_vec(one_data, x.shape().dims())?;
        let tanh_plus_one = add(&one, &tanh_result)?;

        // Compute 0.5 * x * (1 + tanh_result)
        let half = T::from(0.5).unwrap();
        let x_scaled = mul(x, &tanh_plus_one)?;
        let result = x_scaled.mul_scalar(half)?;

        Ok(result)
    }
}

impl<B, S, T> Default for GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Activation<B, S, T> for GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        GeLU::<B, S, T>::forward(self, x)
    }
}

impl<B, S, T> Module<B, S, T> for GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "GeLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
