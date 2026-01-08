//! 2D Convolution operations for neural networks.

use crate::core::error::{NNError, Result};
use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
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
pub fn conv2d<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + FloatExt + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    if weight_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!(
                "Weight must be 4D [C_out, C_in, K_h, K_w], got {}D",
                weight_shape.len()
            ),
        });
    }

    let (batch_size, in_channels, in_height, in_width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (out_channels, weight_in_channels, kernel_height, kernel_width) = (
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
    );

    if in_channels != weight_in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "conv2d".to_string(),
            expected: vec![out_channels, in_channels, kernel_height, kernel_width],
            actual: weight_shape.to_vec(),
        });
    }

    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));

    let (out_height, out_width) = conv2d_output_size(
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
    );

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    let output_size = batch_size * out_channels * out_height * out_width;
    let mut output_data = vec![T::from(0.0).unwrap(); output_size];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = T::from(0.0).unwrap();

                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            if ih >= padding_h
                                && ih < in_height + padding_h
                                && iw >= padding_w
                                && iw < in_width + padding_w
                            {
                                let input_ih = ih - padding_h;
                                let input_iw = iw - padding_w;

                                for ic in 0..in_channels {
                                    let input_idx = ((b * in_channels + ic) * in_height + input_ih)
                                        * in_width
                                        + input_iw;
                                    let weight_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                        * kernel_width
                                        + kw;

                                    sum = sum + input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }

                    if let Some(bias_tensor) = bias {
                        sum = sum + bias_tensor.as_slice()[oc];
                    }

                    let output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }

    let output_shape = vec![batch_size, out_channels, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

/// Applies a 2D transposed convolution (deconvolution) over an input signal.
pub fn conv2d_transpose<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    output_padding: Option<(usize, usize)>,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + FloatExt + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    if weight_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!(
                "Weight must be 4D [C_in, C_out, K_h, K_w], got {}D",
                weight_shape.len()
            ),
        });
    }

    let (batch_size, in_channels, in_height, in_width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (weight_in_channels, out_channels, kernel_height, kernel_width) = (
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
    );

    if in_channels != weight_in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "conv2d_transpose".to_string(),
            expected: vec![
                weight_in_channels,
                out_channels,
                kernel_height,
                kernel_width,
            ],
            actual: weight_shape.to_vec(),
        });
    }

    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));
    let (output_padding_h, output_padding_w) = output_padding.unwrap_or((0, 0));

    let out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    let output_size = batch_size * out_channels * out_height * out_width;
    let mut output_data = vec![T::from(0.0).unwrap(); output_size];

    for b in 0..batch_size {
        for ic in 0..in_channels {
            for ih in 0..in_height {
                for iw in 0..in_width {
                    let input_val =
                        input_data[((b * in_channels + ic) * in_height + ih) * in_width + iw];

                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            #[allow(clippy::needless_range_loop)]
                            for oc in 0..out_channels {
                                let oh = ih * stride_h + kh;
                                let ow = iw * stride_w + kw;

                                if oh < out_height && ow < out_width {
                                    let weight_idx = ((ic * out_channels + oc) * kernel_height
                                        + kh)
                                        * kernel_width
                                        + kw;
                                    let output_idx = ((b * out_channels + oc) * out_height + oh)
                                        * out_width
                                        + ow;

                                    output_data[output_idx] = output_data[output_idx]
                                        + input_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(bias_tensor) = bias {
        let bias_data = bias_tensor.as_slice();
        for b in 0..batch_size {
            #[allow(clippy::needless_range_loop)]
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let output_idx =
                            ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        output_data[output_idx] = output_data[output_idx] + bias_data[oc];
                    }
                }
            }
        }
    }

    let output_shape = vec![batch_size, out_channels, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}
