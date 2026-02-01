use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, U8};
use storage::Storage;

/// Returns a new 1-D tensor which indexes the input tensor according to the boolean mask.
///
/// # Arguments
/// * `tensor` - The input tensor.
/// * `mask` - The boolean mask.
pub fn masked_select<B, S, T, MB, MS>(
    tensor: &Tensor<B, S, T>,
    mask: &Tensor<MB, MS, U8>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T> + crate::ops::TensorStorageOps<T>,
    T: DataType + Copy,
    MB: Backend<Data = U8> + Clone,
    MS: Storage<U8> + Clone + crate::ops::TensorStorageOps<U8> + storage::StorageFromVec<U8>,
{

    // Fallback CPU implementation:
    // Convert to dense to access data.
    let dense_tensor = tensor.to_dense_generic()?;
    // We assume mask can be converted to dense bool tensor.
    // Given the generic signature S: Storage<bool>, to_dense_generic might fail if S doesn't support generic T.
    // however, provided S handles it, we get DenseStorage<bool>.
    let mask_dense = mask.to_dense_generic()?;
    
    let t_data = dense_tensor.as_slice();
    let m_data = mask_dense.as_slice();
    
    if t_data.len() != m_data.len() {
         return Err(crate::TensorError::ShapeMismatch {
            expected: dense_tensor.shape().dims().to_vec(),
            actual: mask_dense.shape().dims().to_vec(),
            operation: "masked_select",
        });
    }
    
    let mut result_data = Vec::with_capacity(t_data.len());
    for (val, &keep) in t_data.iter().zip(m_data.iter()) {
        if keep.0 != 0 {
            result_data.push(*val);
        }
    }
    
    let len = result_data.len();
    Tensor::from_vec_with_backend(result_data, &[len], tensor.backend.clone())
}

/// Fills elements of self tensor with value where mask is True (non-zero).
pub fn masked_fill<B, S, T, MB, MS>(
    tensor: &Tensor<B, S, T>,
    mask: &Tensor<MB, MS, U8>,
    value: T,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T> + crate::ops::TensorStorageOps<T>,
    T: DataType + Copy,
    MB: Backend<Data = U8> + Clone,
    MS: Storage<U8> + Clone + crate::ops::TensorStorageOps<U8> + storage::StorageFromVec<U8>,
{

    // Create tensor filled with value
    // We can use from_vec with repeated value
    let numel = tensor.numel();
    let val_data = vec![value; numel];
    let value_tensor = Tensor::from_vec_with_backend(val_data, tensor.shape().dims(), tensor.backend.clone())?;
    
    // where(mask, value, tensor)
    crate::ops::comparison::where_cond::where_cond(mask, &value_tensor, tensor)
}

