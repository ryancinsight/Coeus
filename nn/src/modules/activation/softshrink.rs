//! Softshrink Activation Function
//!
//! Softshrink(x, λ) = { x - λ, if x > λ
//!                   { x + λ, if x < -λ  
//!                   { 0,     otherwise

use crate::core::error::Result;
use crate::modules::activation::Activation;
use crate::{Module, Parameter};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

/// Softshrink activation function
#[derive(Clone, Debug)]
pub struct Softshrink<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    /// Lambda parameter (threshold)
    pub lambd: T,
    _marker: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Softshrink<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    /// Create a new Softshrink with default lambda=0.5
    pub fn new() -> Self {
        Self {
            lambd: T::from_f64(0.5).unwrap_or_else(|| T::zero()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new Softshrink with custom lambda
    pub fn with_lambd(lambd: T) -> Self {
        Self {
            lambd,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Default for Softshrink<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Activation<B, S, T> for Softshrink<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Copy + PartialOrd + Send + Sync + 'static,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let data = x.as_slice();
        let lambd = self.lambd;
        let neg_lambd = -lambd;
        
        let result: Vec<T> = data
            .iter()
            .map(|&v| {
                if v > lambd {
                    v - lambd
                } else if v < neg_lambd {
                    v + lambd
                } else {
                    T::zero()
                }
            })
            .collect();

        Tensor::from_vec_with_backend(result, x.shape().dims(), x.backend().clone())
            .map_err(Into::into)
    }
}

impl<B, S, T> Module<B, S, T> for Softshrink<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Copy + PartialOrd + Send + Sync + 'static,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        <Self as Activation<B, S, T>>::forward(self, input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "Softshrink"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;

    #[test]
    fn test_softshrink_basic() {
        let softshrink = Softshrink::<Float32>::new();
        let tensor: Tensor<TestBackend, TestStorage, Float32> =
            Tensor::from_vec(vec![Float32(-1.0), Float32(0.0), Float32(1.0)], &[3]).unwrap();
        
        let result = softshrink.forward(&tensor).unwrap();
        let data = result.as_slice();
        
        // x=-1.0: -1.0 < -0.5, so output = -1.0 + 0.5 = -0.5
        assert!((data[0].0 - (-0.5)).abs() < 1e-5);
        // x=0.0: within threshold, output = 0
        assert!((data[1].0 - 0.0).abs() < 1e-5);
        // x=1.0: 1.0 > 0.5, so output = 1.0 - 0.5 = 0.5
        assert!((data[2].0 - 0.5).abs() < 1e-5);
    }
}
