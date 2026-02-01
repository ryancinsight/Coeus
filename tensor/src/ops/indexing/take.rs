use crate::ops::dispatch::traits::TensorStorageOps;
use crate::{Result, Tensor};
use backend::Backend;
use storage::Storage;
use crate::tensor_core::StorageToDense;

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + TensorStorageOps<T> + StorageToDense<T>,
    T: dtype::DataType,
{
    /// Select values from the tensor using the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A LongTensor containing indices to select.
    ///
    /// # Returns
    ///
    /// A new Tensor containing the selected values.
    pub fn take<B2, S2>(&self, indices: &Tensor<B2, S2, dtype::int::Int64>) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
    where
        B2: Backend<Data = dtype::int::Int64>,
        S2: Storage<dtype::int::Int64> + TensorStorageOps<dtype::int::Int64> + StorageToDense<dtype::int::Int64>,
    {
        // For now, convert everything to dense to use the backend primitive
        // TODO: Adding Strided support to backend take/put would allow optimizing this
        let self_dense = self.storage().to_dense()?;
        let indices_dense = indices.storage().to_dense()?;

        let result_storage = self.backend().take_dense(&self_dense, &indices_dense)?;
        
        Ok(Tensor::from_storage(result_storage, self.backend().clone()))
    }
}

pub fn take<B, S, T, B2, S2>(
    tensor: &Tensor<B, S, T>,
    indices: &Tensor<B2, S2, dtype::int::Int64>,
) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + TensorStorageOps<T> + StorageToDense<T>,
    T: dtype::DataType + num_traits::Num + Copy,
    B2: Backend<Data = dtype::int::Int64>,
    S2: Storage<dtype::int::Int64> + TensorStorageOps<dtype::int::Int64> + StorageToDense<dtype::int::Int64>,
{
    tensor.take(indices)
}
