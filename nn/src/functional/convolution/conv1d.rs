//! 1D Convolution operations for neural networks.

use crate::core::error::Result;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

/// Compute output length for 1D convolution.
pub fn conv1d_output_size(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    (input_length + 2 * padding - kernel_size) / stride + 1
}

/// Perform 1D convolution using functional API.
/// Perform 1D convolution using functional API.
pub fn conv1d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    Ok(tensor::ops::conv::conv1d(input, weight, bias, stride, padding)?)
}
