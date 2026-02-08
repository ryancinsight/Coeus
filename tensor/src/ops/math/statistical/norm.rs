//! Tensor norm operations

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Matrix or vector norm
pub fn norm<T, B, S>(tensor: &Tensor<B, S, T>, p: T, dim: Option<usize>, keepdim: bool) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + dtype::traits::FloatExt + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + storage::StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    // Simplified global norm for now to satisfy build
    if dim.is_none() {
        let sum_sq: T = tensor.as_slice().iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
        let val = sum_sq.powf(p.recip());
        return Tensor::from_vec_with_backend(vec![val], vec![1], tensor.backend.clone());
    }
    
    // Dim-wise norm would go here
    todo!("Dimension-wise norm not yet fully restored in refactor")
}
