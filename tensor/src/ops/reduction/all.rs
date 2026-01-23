//! Logical all reduction operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Tests if all elements along specified dimensions evaluate to true.
///
/// For numeric tensors, any non-zero value is considered true.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dims` - Dimensions to reduce over. None means all dimensions.
/// * `keepdim` - Whether to keep the reduced dimensions
pub fn all<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + PartialEq + num_traits::Zero + num_traits::One + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
{
    let data = tensor.as_slice();
    let shape = tensor.shape().dims();
    
    match dims {
        None => {
            // Reduce all dimensions
            let result = if data.iter().all(|&x| x != T::zero()) {
                T::one()
            } else {
                T::zero()
            };
            let out_shape = if keepdim { vec![1; shape.len()] } else { vec![1] };
            Tensor::from_vec_with_backend(vec![result], &out_shape, tensor.backend.clone())
        }
        Some(dims_arr) => {
            // For simplicity, implement full reduction first
            // A proper dim-wise reduction would require more complex logic
            if dims_arr.is_empty() || dims_arr.len() == shape.len() {
                let result = if data.iter().all(|&x| x != T::zero()) {
                    T::one()
                } else {
                    T::zero()
                };
                let out_shape = if keepdim { vec![1; shape.len()] } else { vec![1] };
                Tensor::from_vec_with_backend(vec![result], &out_shape, tensor.backend.clone())
            } else {
                Err(TensorError::UnsupportedOperation {
                    operation: "all (partial dim)".to_string(),
                    storage_type: "partial dimension reduction not yet implemented".to_string(),
                })
            }
        }
    }
}
