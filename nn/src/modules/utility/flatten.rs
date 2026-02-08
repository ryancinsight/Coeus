use crate::core::error::Result;
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

/// Flatten module
#[derive(Debug, Clone)]
pub struct Flatten<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    start_dim: isize,
    end_dim: isize,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> Flatten<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    pub fn new(start_dim: isize, end_dim: isize) -> Self {
        Self {
            start_dim,
            end_dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for Flatten<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::FromPrimitive + num_traits::Zero + num_traits::One + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let flattened = input.flatten(self.start_dim.try_into().unwrap(), self.end_dim.try_into().unwrap())?;
        // Convert to generic storage type S
        // This handles cases where flatten returns StridedStorage but Module expects S
        let dense = flattened.to_dense_generic()?;
        let storage = S::from_vec(dense.as_slice().to_vec(), dense.shape().dims())?;
        Ok(Tensor::from_storage(storage, input.backend().clone()))
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "Flatten"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
