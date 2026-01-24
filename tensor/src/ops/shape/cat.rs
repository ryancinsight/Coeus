//! Concatenation operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::CatFunction;

/// Concatenates the given sequence of tensors along the given dimension.
pub fn cat<B, T, S>(
    tensors: &[Tensor<B, S, T>],
    dim: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    if tensors.is_empty() {
        return Err(TensorError::EmptyTensor);
    }

    let first_shape = tensors[0].shape().dims();
    let ndim = first_shape.len();

    if dim >= ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: dim,
            message: format!("Dimension out of range for cat: dim={}", dim),
        });
    }

    let mut cat_dim_size = 0;
    for (_i, t) in tensors.iter().enumerate() {
        let shape = t.shape().dims();
        if shape.len() != ndim {
            return Err(TensorError::ShapeMismatch {
                expected: first_shape.to_vec(),
                actual: shape.to_vec(),
                operation: "cat",
            });
        }
        for j in 0..ndim {
            if j != dim && shape[j] != first_shape[j] {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    actual: shape.to_vec(),
                    operation: "cat",
                });
            }
        }
        cat_dim_size += shape[dim];
    }

    let mut out_shape = first_shape.to_vec();
    out_shape[dim] = cat_dim_size;

    let total_elements: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(total_elements);

    // Calculate slicing parameters
    let outer_size: usize = first_shape[..dim].iter().product();
    let inner_size: usize = first_shape[dim + 1..].iter().product();

    // Iterate through outer fragments, then each tensor's contribution to the cat dim, then inner fragments
    for outer in 0..outer_size {
        for t in tensors {
            let t_dense = t.to_dense_generic()?;
            let t_data = t_dense.as_slice();
            let t_dim_size = t.shape().dims()[dim];
            
            let start = outer * t_dim_size * inner_size;
            let end = start + t_dim_size * inner_size;
            result_data.extend_from_slice(&t_data[start..end]);
        }
    }

    let mut result = Tensor::from_vec_with_backend(result_data, &out_shape, tensors[0].backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensors.iter().any(|t| t.requires_grad()) {
        let arc_inputs: Vec<Arc<Tensor<B, S, T>>> = tensors.iter().map(|t| Arc::new(t.clone())).collect();
        let grad_fn = CatFunction::new(arc_inputs, dim);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
