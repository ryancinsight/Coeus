use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, DenseStorage, Result, Storage,
    StorageFromVec, Tensor,
};

pub trait ArithmeticDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
    fn dispatch_add(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
    fn dispatch_mul(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>;
}

impl<B, S, T> ArithmeticDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn dispatch_add(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_add(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_mul(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_mul(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let sum_value = input.storage.storage_sum(self)?;
        let scalar_data = vec![sum_value];
        let scalar_storage = DenseStorage::from_vec(scalar_data, &[1]).map_err(crate::TensorError::StorageError)?;
        Ok(Tensor::from_storage(scalar_storage, self.clone()))
    }
}
