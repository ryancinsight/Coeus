//! Squeeze operation - removes dimensions of size 1

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Removes dimensions of size 1 from the tensor shape.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Specific dimension to squeeze, or None to squeeze all size-1 dims
pub fn squeeze<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dim: Option<isize>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Clone + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
{
    let shape = tensor.shape().dims();
    let ndim = shape.len();
    
    let new_shape: Vec<usize> = match dim {
        None => {
            // Squeeze all dimensions of size 1
            shape.iter().cloned().filter(|&d| d != 1).collect()
        }
        Some(d) => {
            // Normalize negative dimension
            let d_normalized = if d < 0 {
                (ndim as isize + d) as usize
            } else {
                d as usize
            };
            
            if d_normalized >= ndim {
                return Err(TensorError::ShapeError {
                    expected: ndim,
                    actual: d_normalized,
                    message: format!("Dimension {d} out of range for tensor with {ndim} dimensions"),
                });
            }
            
            // Only squeeze if dimension is 1
            if shape[d_normalized] == 1 {
                let mut new_shape = shape.to_vec();
                new_shape.remove(d_normalized);
                new_shape
            } else {
                shape.to_vec()
            }
        }
    };
    
    // Handle edge case: empty shape
    let final_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
    
    // Create new tensor with same data, different shape
    let data = tensor.as_slice().to_vec();
    Tensor::from_vec_with_backend(data, &final_shape, tensor.backend.clone())
}
