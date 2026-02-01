use crate::Tensor;
use crate::tensor_backend_dispatch::{TensorBackendDispatcher, TensorDispatcher};
use crate::ops::TensorStorageOps;
use storage::{Storage, StorageFromVec};
use backend::Backend;
use dtype::DataType;
use crate::Result;

/// Applies a 1D max pooling over an input signal composed of several input planes.
pub fn max_pool1d<B, S, T>(
    input: &Tensor<B, S, T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
{
    TensorDispatcher::max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
}

/// Applies a 2D max pooling over an input signal composed of several input planes.
pub fn max_pool2d<B, S, T>(
    input: &Tensor<B, S, T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
{
    TensorDispatcher::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
}

/// Applies a 3D max pooling over an input signal composed of several input planes.
pub fn max_pool3d<B, S, T>(
    input: &Tensor<B, S, T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    ceil_mode: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
{
    TensorDispatcher::max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
}
