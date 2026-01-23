//! Transpose operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Returns a tensor that is a transposed version of the input.
pub fn transpose<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dim0: usize,
    dim1: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let shape = tensor.shape().dims();
    let ndim = shape.len();

    if dim0 >= ndim || dim1 >= ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: std::cmp::max(dim0, dim1),
            message: format!("Dimension out of range for transpose: dim0={}, dim1={}", dim0, dim1),
        });
    }

    let mut new_shape = shape.to_vec();
    new_shape.swap(dim0, dim1);

    let input_dense = tensor.to_dense_generic()?;
    let input_data = input_dense.as_slice();
    let mut result_data = vec![T::zero(); input_data.len()];

    let mut input_strides = vec![1; ndim];
    for i in (1..ndim).rev() {
        input_strides[i - 1] = input_strides[i] * shape[i];
    }

    let mut output_strides = vec![1; ndim];
    for i in (1..ndim).rev() {
        output_strides[i - 1] = output_strides[i] * new_shape[i];
    }

    // Permutation for index mapping
    let mut perm = (0..ndim).collect::<Vec<_>>();
    perm.swap(dim0, dim1);

    for i in 0..input_data.len() {
        let mut temp_idx = i;
        let mut coords = vec![0; ndim];
        for j in (0..ndim).rev() {
            coords[j] = temp_idx % shape[j];
            temp_idx /= shape[j];
        }

        let mut out_idx = 0;
        for j in 0..ndim {
            out_idx += coords[perm[j]] * output_strides[j];
        }
        result_data[out_idx] = input_data[i];
    }

    let mut result = Tensor::from_vec_with_backend(result_data, &new_shape, tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = crate::functions::layout::TransposeFunction::new(Arc::new(tensor.clone()), dim0, dim1);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

