//! Gather operation - select elements along dimension using index tensor
//!
//! Equivalent to PyTorch's `torch.gather(input, dim, index)`.

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Gathers values along an axis specified by dim.
///
/// For a 3-D tensor the output is specified by:
/// ```text
/// out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
/// out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
/// out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
/// ```
///
/// # Arguments
/// * `input` - The source tensor
/// * `dim` - The axis along which to index
/// * `index` - The indices of elements to gather
pub fn gather<B, T, S>(
    input: &Tensor<B, S, T>,
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
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

    // Validate index shape matches input shape except for gathered dimension
    if index_shape.len() != ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: index_shape.len(),
            message: "Index must have same number of dimensions as input".to_string(),
        });
    }

    for (i, (&idx_dim, &inp_dim)) in index_shape.iter().zip(input_shape.iter()).enumerate() {
        if i != dim && idx_dim != inp_dim {
            return Err(TensorError::ShapeError {
                expected: inp_dim,
                actual: idx_dim,
                message: format!("Index shape mismatch at dimension {}", i),
            });
        }
    }

    let input_data = input.as_slice();
    let output_numel: usize = index_shape.iter().product();
    let mut output_data = alloc::vec![T::default(); output_numel];

    // Compute strides for input tensor
    let mut input_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Compute strides for index/output tensor
    let mut output_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        output_strides[i] = output_strides[i + 1] * index_shape[i + 1];
    }

    // Iterate over all output positions
    for out_idx in 0..output_numel {
        // Convert flat index to multi-dimensional index
        let mut coords = alloc::vec![0usize; ndim];
        let mut remaining = out_idx;
        for d in 0..ndim {
            coords[d] = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // Replace coordinate at dim with the index value
        let gather_idx = index[out_idx];
        if gather_idx >= input_shape[dim] {
            return Err(TensorError::ShapeError {
                expected: input_shape[dim],
                actual: gather_idx,
                message: format!("Index {} out of bounds for dimension {} with size {}", 
                    gather_idx, dim, input_shape[dim]),
            });
        }
        coords[dim] = gather_idx;

        // Compute flat input index
        let input_flat_idx: usize = coords.iter().zip(input_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        output_data[out_idx] = input_data[input_flat_idx];
    }

    Tensor::from_vec_with_backend(output_data, index_shape, input.backend.clone())
}
