//! Scatter operation - write values into tensor at specified indices
//!
//! Equivalent to PyTorch's `tensor.scatter_(dim, index, src)`.

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Writes all values from the tensor src into self at the indices specified in the index tensor.
///
/// For a 3-D tensor with dim=0:
/// ```text
/// self[index[i][j][k]][j][k] = src[i][j][k]
/// ```
///
/// # Arguments
/// * `input` - The destination tensor (values are scattered into this)
/// * `dim` - The axis along which to index
/// * `index` - The indices to scatter into
/// * `src` - The source tensor with values to scatter
pub fn scatter<B, T, S>(
    input: &Tensor<B, S, T>,
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<B, S, T>,
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

    // Validate shapes
    if index_shape.len() != ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: index_shape.len(),
            message: "Index must have same number of dimensions as input".to_string(),
        });
    }

    let src_shape = src.shape().dims();
    if src_shape != index_shape {
        return Err(TensorError::ShapeMismatch {
            expected: index_shape.to_vec(),
            actual: src_shape.to_vec(),
            operation: "scatter",
        });
    }

    // Start with copy of input data
    let mut output_data = input.as_slice().to_vec();
    let src_data = src.as_slice();

    // Compute strides for input/output tensor
    let mut input_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Compute strides for index/src tensor
    let mut src_strides = alloc::vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        src_strides[i] = src_strides[i + 1] * index_shape[i + 1];
    }

    let numel: usize = index_shape.iter().product();

    // Iterate over all source positions and scatter
    for src_idx in 0..numel {
        // Convert flat index to multi-dimensional index
        let mut coords = alloc::vec![0usize; ndim];
        let mut remaining = src_idx;
        for d in 0..ndim {
            coords[d] = remaining / src_strides[d];
            remaining %= src_strides[d];
        }

        // Get scatter destination index
        let scatter_idx = index[src_idx];
        if scatter_idx >= input_shape[dim] {
            return Err(TensorError::ShapeError {
                expected: input_shape[dim],
                actual: scatter_idx,
                message: format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    scatter_idx, dim, input_shape[dim]
                ),
            });
        }
        coords[dim] = scatter_idx;

        // Compute flat output index
        let output_flat_idx: usize = coords
            .iter()
            .zip(input_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        output_data[output_flat_idx] = src_data[src_idx];
    }

    Tensor::from_vec_with_backend(output_data, input_shape, input.backend.clone())
}
