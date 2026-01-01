//! General tensor manipulation operations.
//!
//! This module contains tensor operations that manipulate tensor structure,
//! concatenation, splitting, and other structural transformations.

use crate::Tensor;

/// Concatenates tensors along a specified dimension.
///
/// # Arguments
/// * `tensors` - Slice of tensors to concatenate
/// * `dim` - Dimension along which to concatenate (0-based)
///
/// # Returns
/// A new tensor with the concatenated result
///
/// # Errors
/// Returns error if tensors have incompatible shapes or concatenation fails
///
/// # Note
/// All input tensors must have the same shape except for the concatenation dimension.
/// For example, concatenating along dimension 0 requires all tensors to have
/// identical shape[1..] but can have different shape[0].
pub fn concatenate_tensors<B, S, T>(
    tensors: &[Tensor<B, S, T>],
    dim: usize,
) -> crate::Result<Tensor<B, S, T>>
where
    B: crate::Backend<Data = T> + Clone,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T>,
    T: crate::DataType,
{
    if tensors.is_empty() {
        return Err(crate::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: "Cannot concatenate empty tensor list".to_string(),
        });
    }

    // Check all tensors have compatible shapes
    let first_shape = tensors[0].shape().dims();
    if dim >= first_shape.len() {
        return Err(crate::TensorError::ShapeError {
            expected: 0,
            actual: dim,
            message: format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                first_shape.len()
            ),
        });
    }

    // Verify all tensors have compatible shapes (same size in all dimensions except dim)
    for (i, tensor) in tensors.iter().enumerate() {
        let shape = tensor.shape().dims();
        if shape.len() != first_shape.len() {
            return Err(crate::TensorError::ShapeError {
                expected: first_shape.len(),
                actual: shape.len(),
                message: format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    shape.len(),
                    first_shape.len()
                ),
            });
        }

        for (j, (&actual, &expected)) in shape.iter().zip(first_shape).enumerate() {
            if j != dim && actual != expected {
                return Err(crate::TensorError::ShapeError {
                    expected,
                    actual,
                    message: format!(
                        "Tensor {} dimension {} has size {}, expected {}",
                        i, j, actual, expected
                    ),
                });
            }
        }
    }

    // Calculate output shape
    let mut output_shape = first_shape.to_vec();
    let total_dim_size: usize = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
    output_shape[dim] = total_dim_size;

    // Calculate total number of elements
    let total_elements: usize = output_shape.iter().product();

    // Concatenate the data
    let mut concatenated_data = vec![T::default(); total_elements];

    // For now, implement basic concatenation by copying data in the correct order
    // This is a simplified implementation - in practice, we'd want more efficient methods
    // for different backends and storage types

    let mut offsets = vec![0; output_shape.len()];
    for tensor in tensors {
        let tensor_shape = tensor.shape().dims();
        let tensor_size = tensor_shape.iter().product::<usize>();

        // Copy this tensor's data with proper index calculation
        for linear_idx in 0..tensor_size {
            // Convert linear index to multi-dimensional coordinates
            let mut coords = vec![0; tensor_shape.len()];
            let mut remaining = linear_idx;
            for (i, &dim_size) in tensor_shape.iter().enumerate().rev() {
                coords[i] = remaining % dim_size;
                remaining /= dim_size;
            }

            // Apply offset for concatenation dimension
            coords[dim] += offsets[dim];

            let mut output_linear_idx = 0;
            let mut multiplier = 1;
            for (i, &coord) in coords.iter().enumerate().rev() {
                output_linear_idx += coord * multiplier;
                multiplier *= output_shape[i];
            }

            concatenated_data[output_linear_idx] = tensor.as_slice()[linear_idx];
        }

        // Update offset for next tensor
        offsets[dim] += tensor_shape[dim];
    }

    Tensor::from_vec(concatenated_data, &output_shape)
}
