use crate::{Tensor, Backend, Storage, DenseStorage, Result, CpuBackend};
use dtype::DataType;
use storage::{StorageToDense, StorageFromVec};

/// Casts a tensor to a different data type.
///
/// Currently performs the cast on CPU by converting to dense storage.
///
/// # Type Parameters
/// * `T2` - The target data type.
/// * `B` - The source backend type.
/// * `S` - The source storage type.
/// * `T` - The source data type.
pub fn cast<T2, B, S, T>(tensor: &Tensor<B, S, T>) -> Result<Tensor<CpuBackend<T2>, DenseStorage<T2>, T2>>
where
    T2: DataType,
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    // 1. Convert to dense (CPU accessible usually, or allows easy iteration)
    let dense = tensor.to_dense_generic()?;
    
    // 2. Perform cast
    let data = dense.as_slice();
    // Use explicit type for closure argument to avoid inference errors
    let cast_data: Result<Vec<T2>> = data.iter()
        .map(|x: &T| x.checked_cast_to::<T2>().map_err(|e| crate::TensorError::InvalidInput { message: format!("Cast error: {:?}", e) }))
        .collect();
    
    let new_data = cast_data?;
    
    // 3. Create new storage
    // Use tensor.storage.shape() to avoid Tensor::shape() trait bound issues if any (though we added StorageFromVec now)
    let shape = tensor.storage.shape();
    let new_storage = DenseStorage::from_vec(new_data, shape.dims())?;
    
    // 4. Create new backend
    let new_backend = CpuBackend::new();
    
    Ok(Tensor::from_storage(new_storage, new_backend))
}
