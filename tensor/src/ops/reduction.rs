//! Reduction operations

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use coeus_backend::{Backend, CpuBackend};

/// Sum along specified dimension
pub fn sum_dim<T: Dtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>, dim: usize) -> Result<Tensor<T, B>> {
    if dim >= tensor.ndim() {
        return Err(TensorError::InvalidOperation {
            message: format!(
                "Dimension {} out of bounds for {}D tensor",
                dim,
                tensor.ndim()
            ),
        });
    }
    if tensor.numel() == 0 {
        return Err(TensorError::InvalidOperation {
            message: "Cannot sum empty tensor".to_string(),
        });
    }

    let mut new_shape = tensor.shape().to_vec();
    new_shape.remove(dim);

    let mut result_data = vec![T::zero(); new_shape.iter().product()];

    // Compute sum along dimension
    let mut result_idx = 0;
    let total_elements = tensor.numel();
    let stride = tensor.shape()[dim];

    for i in 0..total_elements {
        let coords = index_to_coords(i, tensor.shape());
        if coords[dim] == 0 {
            // Start of a new sum group
            let mut sum = T::zero();
            for offset in 0..stride {
                let mut coords_copy = coords.clone();
                coords_copy[dim] = offset;
                let idx = coords_to_index(&coords_copy, tensor.shape());
                sum = sum + tensor.data()[idx];
            }
            result_data[result_idx] = sum;
            result_idx += 1;
        }
    }

    let backend = tensor.backend().clone();
    let mut result = Tensor::from_vec(backend, result_data, new_shape)?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
        // Backward: gradient flows back to input with broadcasting
        // For sum, gradient is broadcasted to input shape
        // Edge case: empty tensor handled by zero gradient
    }

    Ok(result)
}

/// Sum of all elements
pub fn sum<T: Dtype, B: Backend<T> + Clone>(tensor: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    if tensor.numel() == 0 {
        return Err(TensorError::InvalidOperation {
            message: "Cannot sum empty tensor".to_string(),
        });
    }

    let sum_val = tensor.data().iter().fold(T::zero(), |acc, x| acc + *x);
    let backend = tensor.backend().clone();
    let mut result = Tensor::from_vec(backend, vec![sum_val], vec![])?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
        // Backward: gradient is ones_like(input) for sum
    }

    Ok(result)
}

/// Mean along specified dimension
pub fn mean_dim<T: FloatDtype, B: Backend<T> + Clone + Send + Sync>(tensor: &Tensor<T, B>, dim: usize) -> Result<Tensor<T, B>>
where
    CpuBackend: Backend<T>,
{
    let sum = sum_dim(tensor, dim)?;
    let count = tensor.shape()[dim] as f64;

    let data = sum
        .data()
        .iter()
        .map(|x| *x / T::from(count).unwrap())
        .collect();

    let backend = sum.backend().clone();
    let mut result = Tensor::from_vec(backend, data, sum.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
        // Backward: gradient flows back to input with broadcasting
        // For sum, gradient is broadcasted to input shape
        // Edge case: empty tensor handled by zero gradient
    }

    Ok(result)
}

/// Concatenate tensors along specified dimension
pub fn cat<T: FloatDtype, B: Backend<T> + Clone + Send + Sync>(tensors: &[&Tensor<T, B>], dim: usize) -> Result<Tensor<T, B>>
where
    CpuBackend: Backend<T>,
{
    if tensors.is_empty() {
        return Err(TensorError::InvalidOperation {
            message: "Cannot concatenate empty tensor list".to_string(),
        });
    }

    // Check that all tensors have the same shape except for the concatenation dimension
    let first_shape = tensors[0].shape();
    for tensor in tensors.iter().skip(1) {
        for (i, (a, b)) in first_shape.iter().zip(tensor.shape()).enumerate() {
            if i != dim && a != b {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }
    }

    // Calculate new shape
    let mut new_shape = first_shape.to_vec();
    new_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();

    let total_size: usize = new_shape.iter().product();
    let mut result_data = vec![T::zero(); total_size];

    // Concatenate data
    let mut offset = 0;
    for tensor in tensors {
        let size = tensor.numel();
        result_data[offset..offset + size].copy_from_slice(tensor.data());
        offset += size;
    }

    let backend = tensors[0].backend().clone();
    let mut result = Tensor::from_vec(backend, result_data, new_shape)?;

    if tensors.iter().any(|t| t.requires_grad()) {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
    }

    Ok(result)
}

/// Utility function to convert flat index to coordinates
fn index_to_coords(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = index;

    for (i, &dim_size) in shape.iter().enumerate().rev() {
        coords[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    coords
}

/// Utility function to convert coordinates to flat index
fn coords_to_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;

    for (i, &coord) in coords.iter().enumerate().rev() {
        index += coord * stride;
        stride *= shape[i];
    }

    index
}
