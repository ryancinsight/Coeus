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
/// identical shape\[1..\] but can have different shape\[0\].
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

/// Stacks tensors along a new dimension.
///
/// Unlike `concatenate_tensors` which joins tensors along an existing dimension,
/// `stack_tensors` creates a new dimension and places each input tensor along it.
///
/// # Arguments
/// * `tensors` - Slice of tensors to stack (must all have identical shapes)
/// * `dim` - Dimension at which to insert the new stacking dimension (0 to ndim inclusive)
///
/// # Returns
/// A new tensor with shape [..., N, ...] where N is the number of input tensors
/// and the new dimension is inserted at position `dim`.
///
/// # Errors
/// Returns error if tensors have incompatible shapes or if dim is out of bounds.
///
/// # Examples
/// ```ignore
/// // Stack 3 tensors of shape [2, 3] along dim=0
/// // Result has shape [3, 2, 3]
/// let stacked = stack_tensors(&[a, b, c], 0)?;
///
/// // Stack along dim=1
/// // Result has shape [2, 3, 3]
/// let stacked = stack_tensors(&[a, b, c], 1)?;
/// ```
pub fn stack_tensors<B, S, T>(
    tensors: &[Tensor<B, S, T>],
    dim: usize,
) -> crate::Result<Tensor<B, S, T>>
where
    B: crate::Backend<Data = T> + Clone + Default,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T>,
    T: crate::DataType,
{
    if tensors.is_empty() {
        return Err(crate::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: "Cannot stack empty tensor list".to_string(),
        });
    }

    let first_shape = tensors[0].shape().dims();
    let num_tensors = tensors.len();

    // dim can be from 0 to ndim (inclusive) for stacking (adding a new dimension)
    if dim > first_shape.len() {
        return Err(crate::TensorError::ShapeError {
            expected: first_shape.len(),
            actual: dim,
            message: format!(
                "Dimension {} out of bounds for stacking tensor with {} dimensions",
                dim,
                first_shape.len()
            ),
        });
    }

    // Verify all tensors have identical shapes
    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        let shape = tensor.shape().dims();
        if shape != first_shape {
            return Err(crate::TensorError::ShapeError {
                expected: first_shape.len(),
                actual: shape.len(),
                message: format!(
                    "Tensor {} has shape {:?}, expected {:?}",
                    i, shape, first_shape
                ),
            });
        }
    }

    // Calculate output shape: insert num_tensors at dimension dim
    let mut output_shape = first_shape.to_vec();
    output_shape.insert(dim, num_tensors);

    let tensor_elements: usize = first_shape.iter().product();
    let total_elements = tensor_elements * num_tensors;
    let mut output_data = vec![T::default(); total_elements];

    // For each element position in input tensors, copy to correct output position
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        let tensor_data = tensor.as_slice();

        for (elem_idx, &value) in tensor_data.iter().enumerate().take(tensor_elements) {
            // Convert elem_idx to multi-dimensional coordinates in input tensor
            let mut coords = vec![0; first_shape.len()];
            let mut remaining = elem_idx;
            for (i, &dim_size) in first_shape.iter().enumerate().rev() {
                coords[i] = remaining % dim_size;
                remaining /= dim_size;
            }

            // Insert tensor_idx at position dim for output coordinates
            coords.insert(dim, tensor_idx);

            // Convert output coordinates to linear index
            let mut output_linear_idx = 0;
            let mut multiplier = 1;
            for (i, &coord) in coords.iter().enumerate().rev() {
                output_linear_idx += coord * multiplier;
                multiplier *= output_shape[i];
            }

            output_data[output_linear_idx] = value;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}
