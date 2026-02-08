use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, U8};
use storage::Storage;

/// Return a tensor of elements selected from either input or other, depending on condition (non-zero=True).
///
/// # Arguments
/// * `condition` - When True (nonzero), yield x, otherwise yield y
/// * `input` - The values selected at indices where condition is True
/// * `other` - The values selected at indices where condition is False
///
/// # Returns
/// A new tensor with the selected values.
pub fn where_cond<B, S, T, MB, MS>(
    condition: &Tensor<MB, MS, U8>,
    input: &Tensor<B, S, T>,
    other: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T> + crate::ops::TensorStorageOps<T>,
    T: DataType,
    MB: Backend<Data = U8> + Clone,
    MS: Storage<U8> + Clone + storage::StorageFromVec<U8> + storage::StorageToDense<U8>,
{

    // Alignment check
    if condition.shape() != input.shape() || input.shape() != other.shape() {
         return Err(crate::TensorError::ShapeMismatch {
            expected: input.shape().dims().to_vec(),
            actual: condition.shape().dims().to_vec(), 
            operation: "where_cond",
        });
    }

    // Convert to dense
    let input_dense = input.to_dense_generic()?;
    let other_dense = other.to_dense_generic()?;
    
    // For condition, we need to handle U8 storage specifically
    let cond_dense = condition.to_dense_generic()?;
    
    let cond_data = cond_dense.as_slice();
    let input_data = input_dense.as_slice(); 
    let other_data = other_dense.as_slice();
    
    let mut result_data = Vec::with_capacity(input_data.len());
    // Safe iteration since we checked shapes (and thus sizes) match
    for i in 0..input_data.len() {
        result_data.push(if cond_data[i].0 != 0 { input_data[i] } else { other_data[i] });
    }
    
    Tensor::from_vec_with_backend(result_data, input.shape().dims(), input.backend.clone())
}
