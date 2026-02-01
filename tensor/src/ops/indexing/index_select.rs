//! Index select operation - select elements along a dimension
//!
//! Equivalent to PyTorch's `torch.index_select(input, dim, index)`.

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Returns a new tensor which indexes the input tensor along dimension dim using
/// the entries in index which is a 1D tensor.
///
/// # Arguments
/// * `input` - The source tensor
/// * `dim` - The dimension to index along
/// * `index` - 1D tensor of indices to select
pub fn index_select<B, T, S>(
    input: &Tensor<B, S, T>,
    dim: usize,
    index: &[usize],
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Copy + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let input_shape = input.shape().dims();
    let ndim = input_shape.len();

    if dim >= ndim {
        return Err(TensorError::InvalidDimension { dim, ndim });
    }

    // Validate indices are in bounds
    for &idx in index {
        if idx >= input_shape[dim] {
            return Err(TensorError::ShapeError {
                expected: input_shape[dim],
                actual: idx,
                message: format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, dim, input_shape[dim]
                ),
            });
        }
    }

    // Compute output shape (same as input but with dim replaced by index length)
    let mut output_shape = input_shape.to_vec();
    output_shape[dim] = index.len();

    let input_data = input.as_slice();
    let output_numel: usize = output_shape.iter().product();
    let mut output_data = alloc::vec![T::default(); output_numel];

    // Compute strides
    let mut input_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    let mut output_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Iterate over all output positions
    for out_idx in 0..output_numel {
        // Convert flat index to multi-dimensional coords
        let mut coords = alloc::vec![0usize; ndim];
        let mut remaining = out_idx;
        for d in 0..ndim {
            coords[d] = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // Replace coordinate at dim with the actual index value
        coords[dim] = index[coords[dim]];

        // Compute flat input index
        let input_flat_idx: usize = coords
            .iter()
            .zip(input_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        output_data[out_idx] = input_data[input_flat_idx];
    }

    Tensor::from_vec_with_backend(output_data, &output_shape, input.backend.clone())
}
