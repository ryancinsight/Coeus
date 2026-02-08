use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, Result, Storage,
    StorageFromVec, Tensor,
};

pub trait ActivationDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Default;
}

impl<B, S, T> ActivationDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Default,
    {
        let result_storage = input.storage.storage_relu(self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }
}
