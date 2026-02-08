use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, Result, Storage,
    StorageFromVec, Tensor,
};

pub trait LinalgDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
    fn dispatch_matmul(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
}

impl<B, S, T> LinalgDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn dispatch_matmul(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_matmul(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }
}
