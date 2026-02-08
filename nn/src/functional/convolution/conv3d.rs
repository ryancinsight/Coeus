//! 3D Convolution operations for neural networks.

use crate::core::error::{NNError, Result};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

/// Pad a 3D tensor with zeros according to padding parameters.
pub fn pad_3d<B, S, T>(
    input: &Tensor<B, S, T>,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 5usize {
        return Err(NNError::ShapeMismatch {
            operation: "pad_3d".to_string(),
            expected: vec![0, 0, 0, 0, 0],
            actual: input_shape.to_vec(),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    let padded_depth = input_depth + 2 * padding_d;
    let padded_height = input_height + 2 * padding_h;
    let padded_width = input_width + 2 * padding_w;

    let input_data = input.as_slice();
    let mut padded_data =
        vec![T::zero(); batch_size * channels * padded_depth * padded_height * padded_width];

    // Copy input data to padded tensor with offset
    for b in 0..batch_size {
        for c in 0..channels {
            for d in 0..input_depth {
                for h in 0..input_height {
                    for w in 0..input_width {
                        let input_idx = (((b * channels + c) * input_depth + d) * input_height + h)
                            * input_width
                            + w;
                        let padded_idx = (((b * channels + c) * padded_depth + (d + padding_d))
                            * padded_height
                            + (h + padding_h))
                            * padded_width
                            + (w + padding_w);
                        padded_data[padded_idx] = input_data[input_idx];
                    }
                }
            }
        }
    }

    let padded_shape = [
        batch_size,
        channels,
        padded_depth,
        padded_height,
        padded_width,
    ];
    Tensor::from_vec(padded_data, &padded_shape).map_err(Into::into)
}

/// Compute output dimensions for 3D convolution.
pub fn conv3d_output_size(
    input: (usize, usize, usize),
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> (u64, u64, u64) {
    let (input_depth, input_height, input_width) = input;
    let (kernel_depth, kernel_height, kernel_width) = kernel;
    let (stride_d, stride_h, stride_w) = stride;
    let (padding_d, padding_h, padding_w) = padding;

    let out_depth = (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1;
    let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
    (out_depth as u64, out_height as u64, out_width as u64)
}

/// Perform 3D convolution using functional API.
/// Perform 3D convolution using functional API.
pub fn conv3d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: Option<(usize, usize, usize)>,
    groups: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let (stride_d, stride_h, stride_w) = stride;
    let (padding_d, padding_h, padding_w) = padding;
    let (dilation_d, dilation_h, dilation_w) = dilation.unwrap_or((1, 1, 1));
    
    if dilation_d != 1 || dilation_h != 1 || dilation_w != 1 {
        return Err(crate::core::error::NNError::NotImplemented {
            operation: "Dilation != 1 not supported in functional conv3d yet".to_string(),
        });
    }
    if groups != 1 {
        return Err(crate::core::error::NNError::NotImplemented {
            operation: "Groups != 1 not supported in functional conv3d yet".to_string(),
        });
    }

    Ok(tensor::ops::conv::conv3d(
        input,
        weight,
        bias,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    )?)
}

/// Perform 3D transposed convolution using functional API.
pub fn conv3d_transpose<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    Ok(tensor::ops::conv::conv_transpose3d(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
    )?)
}
