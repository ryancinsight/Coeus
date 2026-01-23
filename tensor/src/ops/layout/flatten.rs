//! Flatten operation - flattens a contiguous range of dims

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Flattens input tensor by reshaping a contiguous range of dimensions into a single dimension.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `start_dim` - First dimension to flatten (inclusive)
/// * `end_dim` - Last dimension to flatten (inclusive, can be negative)
pub fn flatten<B, T, S>(
    tensor: &Tensor<B, S, T>,
    start_dim: isize,
    end_dim: isize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Clone + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
{
    let shape = tensor.shape().dims();
    let ndim = shape.len();
    
    if ndim == 0 {
        return Err(TensorError::ShapeError {
            expected: 1,
            actual: 0,
            message: "Cannot flatten scalar tensor".to_string(),
        });
    }
    
    // Normalize negative dimensions
    let start = if start_dim < 0 {
        (ndim as isize + start_dim) as usize
    } else {
        start_dim as usize
    };
    
    let end = if end_dim < 0 {
        (ndim as isize + end_dim) as usize
    } else {
        end_dim as usize
    };
    
    if start > end || end >= ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: end,
            message: format!("Invalid flatten dimensions: start={start_dim}, end={end_dim}"),
        });
    }
    
    // Compute new shape
    let flattened_size: usize = shape[start..=end].iter().product();
    let mut new_shape = Vec::with_capacity(ndim - (end - start));
    new_shape.extend_from_slice(&shape[..start]);
    new_shape.push(flattened_size);
    new_shape.extend_from_slice(&shape[end + 1..]);
    
    // Create new tensor with same data, different shape
    let data = tensor.as_slice().to_vec();
    Tensor::from_vec_with_backend(data, &new_shape, tensor.backend.clone())
}
