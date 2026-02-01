pub mod kernels;

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use crate::ops::dispatch::TensorStorageOps;
use crate::tensor_backend_dispatch::{TensorBackendDispatcher, TensorDispatcher};

// Re-export kernels for use in default dispatcher
pub use kernels::*;

/// 1D Convolution
pub fn conv1d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv1d(input, weight, bias, stride, padding)
}

/// 1D Transposed Convolution
pub fn conv_transpose1d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    dilation: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv_transpose1d(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )
}

/// 2D Convolution
pub fn conv2d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv2d(input, weight, bias, stride_h, stride_w, padding_h, padding_w)
}

/// 2D Transposed Convolution
pub fn conv_transpose2d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_h: usize,
    output_padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv_transpose2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
    )
}

/// 3D Convolution
pub fn conv3d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv3d(
        input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
    )
}

/// 3D Transposed Convolution
pub fn conv_transpose3d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    TensorDispatcher::conv_transpose3d(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
    )
}
