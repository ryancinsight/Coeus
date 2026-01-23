//! Cumulative product along a dimension

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Cumulative product along a dimension.
pub fn cumprod<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
) -> Result<Tensor<B, S, T>> {
    let shape = tensor.shape().dims();
    if dim >= shape.len() {
        return Err(TensorError::InvalidDimension { dim, ndim: shape.len() });
    }

    let mut data = tensor.as_slice().to_vec();
    let stride: usize = shape.iter().skip(dim + 1).product();
    let stride = if stride == 0 { 1 } else { stride };
    let outer_size: usize = shape.iter().take(dim).product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let dim_size = shape[dim];

    for outer in 0..outer_size {
        for inner in 0..stride {
            let base = outer * dim_size * stride + inner;
            let mut accum = T::one();
            for i in 0..dim_size {
                let idx = base + i * stride;
                accum = accum * data[idx];
                data[idx] = accum;
            }
        }
    }

    Tensor::from_vec_with_backend(data, shape, tensor.backend.clone())
}
