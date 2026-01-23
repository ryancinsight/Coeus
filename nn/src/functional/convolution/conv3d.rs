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
pub fn conv3d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 5usize {
        return Err(NNError::InvalidInput {
            message: "Input must be 5D [batch, channels, depth, height, width]".to_string(),
        });
    }

    if weight_shape.len() != 5usize {
        return Err(NNError::InvalidInput {
            message: "Weight must be 5D [out_channels, in_channels, kernel_d, kernel_h, kernel_w]"
                .to_string(),
        });
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];
    let out_channels = weight_shape[0];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    if weight_shape[1] != in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "conv3d".to_string(),
            expected: vec![
                out_channels,
                in_channels,
                kernel_depth,
                kernel_height,
                kernel_width,
            ],
            actual: weight_shape.to_vec(),
        });
    }

    let (stride_d, stride_h, stride_w) = stride;
    let (padding_d, padding_h, padding_w) = padding;

    let (output_depth, output_height, output_width) = (
        (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1,
        (input_height + 2 * padding_h - kernel_height) / stride_h + 1,
        (input_width + 2 * padding_w - kernel_width) / stride_w + 1,
    );

    // Pad input if necessary
    let padded_input = if padding_d > 0 || padding_h > 0 || padding_w > 0 {
        pad_3d(input, padding_d, padding_h, padding_w)?
    } else {
        input.clone()
    };

    let padded_shape = padded_input.shape().dims();
    let padded_depth = padded_shape[2];
    let padded_height = padded_shape[3];
    let padded_width = padded_shape[4];

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_depth * output_height * output_width;
    let mut output_data = vec![T::zero(); output_size];

    let input_data = padded_input.as_slice();
    let weight_data = weight.as_slice();

    // Perform 3D convolution
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = T::zero();

                        // Convolve over input channels and kernel
                        for ic in 0..in_channels {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let id = od * stride_d + kd;
                                        let ih = oh * stride_h + kh;
                                        let iw = ow * stride_w + kw;

                                        if id < padded_depth
                                            && ih < padded_height
                                            && iw < padded_width
                                        {
                                            let input_idx =
                                                (((b * in_channels + ic) * padded_depth + id)
                                                    * padded_height
                                                    + ih)
                                                    * padded_width
                                                    + iw;
                                            let weight_idx =
                                                (((oc * in_channels + ic) * kernel_depth + kd)
                                                    * kernel_height
                                                    + kh)
                                                    * kernel_width
                                                    + kw;
                                            sum = sum
                                                + input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if let Some(bias_tensor) = bias {
                            let bias_data = bias_tensor.as_slice();
                            sum = sum + bias_data[oc];
                        }

                        let output_idx =
                            (((b * out_channels + oc) * output_depth + od) * output_height + oh)
                                * output_width
                                + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }

    let output_shape = [
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width,
    ];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}
