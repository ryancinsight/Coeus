//! Unsqueeze operation - adds a dimension of size 1

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Inserts a dimension of size 1 at the specified position.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Position at which to insert the new dimension (can be negative)
pub fn unsqueeze<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dim: isize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Clone + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
{
    let shape = tensor.shape().dims();
    let ndim = shape.len();
    
    // Normalize negative dimension
    // For unsqueeze, valid range is [-(ndim+1), ndim]
    let d_normalized = if dim < 0 {
        ((ndim + 1) as isize + dim) as usize
    } else {
        dim as usize
    };
    
    if d_normalized > ndim {
        return Err(TensorError::ShapeError {
            expected: ndim + 1,
            actual: d_normalized,
            message: format!("Dimension {dim} out of range for unsqueeze on tensor with {ndim} dimensions"),
        });
    }
    
    // Insert dimension of size 1 at position d_normalized
    let mut new_shape = shape.to_vec();
    new_shape.insert(d_normalized, 1);
    
    // Create new tensor with same data, different shape
    let data = tensor.as_slice().to_vec();
    Tensor::from_vec_with_backend(data, &new_shape, tensor.backend.clone())
}
