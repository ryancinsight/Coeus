//! 2D Convolution operations for neural networks.

use crate::core::error::Result;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

/// Compute output dimensions for 2D convolution.
pub fn conv2d_output_size(
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> (usize, usize) {
    let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
    (out_height, out_width)
}

/// Applies a 2D convolution over an input signal.
/// Applies a 2D convolution over an input signal.
pub fn conv2d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
    groups: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));
    let (dilation_h, dilation_w) = dilation.unwrap_or((1, 1));
    
    if dilation_h != 1 || dilation_w != 1 {
        return Err(crate::core::error::NNError::NotImplemented {
            operation: "Dilation != 1 not supported in functional conv2d yet".to_string(),
        });
    }
    if groups != 1 {
        return Err(crate::core::error::NNError::NotImplemented {
            operation: "Groups != 1 not supported in functional conv2d yet".to_string(),
        });
    }

    Ok(tensor::ops::conv::conv2d(
        input, weight, bias, stride_h, stride_w, padding_h, padding_w,
    )?)
}

/// Applies a 2D transposed convolution (deconvolution) over an input signal.
/// Applies a 2D transposed convolution (deconvolution) over an input signal.
pub fn conv2d_transpose<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    output_padding: Option<(usize, usize)>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));
    let (output_padding_h, output_padding_w) = output_padding.unwrap_or((0, 0));

    Ok(tensor::ops::conv::conv_transpose2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
    )?)
}
