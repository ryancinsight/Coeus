use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

use super::Activation;

/// Softmax activation
#[derive(Debug, Clone)]
pub struct Softmax<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    dim: Option<isize>,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Softmax<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    pub fn new(dim: Option<isize>) -> Self {
        Self {
            dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Activation<B, S, T> for Softmax<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive + num_traits::Zero + num_traits::One + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let dim = self.dim.unwrap_or((x.shape().ndim() - 1) as isize);
        
        let result = tensor::ops::classification::softmax(x, dim as i64).map_err(crate::core::error::NNError::from)?;
        
        // Result is Tensor<B, DenseStorage<T>, T> usually?
        // Check softmax signature in tensor: 
        // pub fn softmax<...>(tensor: &Tensor<B, S, T>, dim: isize, dtype: Option<...>) -> Result<Tensor<B, S, T>>
        // If it preserves S, we are good.
        // But previously we saw manual conversion logic.
        // If it returns S, good. If Dense, convert.
        
        let dense = result.to_dense_generic().map_err(crate::core::error::NNError::from)?;
        let shape = dense.shape().clone();
        let data = dense.as_slice().to_vec();
        
        let storage = S::from_vec(data, shape.dims()).map_err(crate::core::error::NNError::from)?;
        Ok(Tensor::from_storage(storage, x.backend().clone()))
    }
}

impl<B, S, T> Module<B, S, T> for Softmax<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive + num_traits::Zero + num_traits::One + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        <Self as Activation<B, S, T>>::forward(self, input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "Softmax"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
