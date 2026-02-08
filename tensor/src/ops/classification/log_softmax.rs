use crate::Result;
use crate::Tensor;
use crate::ops::classification::softmax;
use crate::ops::math::log;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, StorageFromVec, StorageToDense};

/// LogSoftmax operation
pub fn log_softmax<B, T, S>(input: &Tensor<B, S, T>, dim: i64) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + FloatExt + Clone + 'static,
    S: StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    // Implementation: log(softmax(input))
    // Softmax returns DenseStorage
    let sm = softmax(input, dim)?;
    
    // Log is element-wise. sm is DenseStorage.
    // Ensure log can handle DenseStorage.
    // crate::ops::math::log calls unary_op which dispatches.
    // DenseStorage implements dispatch traits.
    
    // Explicitly reference log function
    let result = log(&sm)?;
    
    Ok(result)
}
