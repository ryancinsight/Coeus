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
    /// Place values into the tensor at the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A LongTensor containing indices to place values at.
    /// * `values` - A Tensor containing values to place.
    /// * `accumulate` - If true, add values to existing elements. If false, replace them.
    ///
    /// # Returns
    ///
    /// A new Tensor with the values placed.
    pub fn put<B2, S2, B3, S3>(
        &self, 
        indices: &Tensor<B2, S2, dtype::int::Int64>, 
        values: &Tensor<B3, S3, T>,
        accumulate: bool
    ) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
    where
        B2: Backend<Data = dtype::int::Int64>,
        S2: Storage<dtype::int::Int64> + TensorStorageOps<dtype::int::Int64> + StorageToDense<dtype::int::Int64>,
        B3: Backend<Data = T>,
        S3: Storage<T> + TensorStorageOps<T> + StorageToDense<T>,
    {
        let mut self_dense = self.storage().to_dense()?;
        let indices_dense = indices.storage().to_dense()?;
        let values_dense = values.storage().to_dense()?;

        self.backend().put_dense(
            &mut self_dense,
            &indices_dense,
            &values_dense,
            accumulate
        )?;
        Ok(Tensor::from_storage(self_dense, self.backend().clone()))
    }
}

pub fn put<B, S, T, B2, S2, B3, S3>(
    tensor: &Tensor<B, S, T>,
    indices: &Tensor<B2, S2, dtype::int::Int64>,
    values: &Tensor<B3, S3, T>,
    accumulate: bool,
) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + TensorStorageOps<T> + StorageToDense<T>,
    T: dtype::DataType + num_traits::Num + Copy,
    B2: Backend<Data = dtype::int::Int64>,
    S2: Storage<dtype::int::Int64> + TensorStorageOps<dtype::int::Int64> + StorageToDense<dtype::int::Int64>,
    B3: Backend<Data = T>,
    S3: Storage<T> + TensorStorageOps<T> + StorageToDense<T>,
{
    tensor.put(indices, values, accumulate)
}
