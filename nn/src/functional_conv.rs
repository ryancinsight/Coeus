//! Convolution operations for neural networks.
//!
//! This module provides stateless convolution operations for spatial feature extraction
//! in convolutional neural networks, optimized with SIMD acceleration.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::error::{NNError, Result};

/// Applies a 2D convolution over an input signal.
///
/// This function performs 2D convolution with optional stride and padding.
/// The convolution is computed directly without im2col transformation for simplicity.
///
/// # Arguments
/// * `input` - Input tensor of shape `(N, C_in, H_in, W_in)`
/// * `weight` - Weight tensor of shape `(C_out, C_in, K_h, K_w)`
/// * `bias` - Optional bias tensor of shape `(C_out,)`
/// * `stride` - Stride for height and width `(stride_h, stride_w)`. Default: (1, 1)
/// * `padding` - Padding for height and width `(pad_h, pad_w)`. Default: (0, 0)
///
/// # Returns
/// Output tensor of shape `(N, C_out, H_out, W_out)` where:
/// - `H_out = (H_in + 2*pad_h - K_h) / stride_h + 1`
/// - `W_out = (W_in + 2*pad_w - K_w) / stride_w + 1`
///
/// # Examples
/// ```rust
/// use nn::functional_conv::conv2d;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 32, 32]).unwrap();
/// let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[64, 3, 3, 3]).unwrap();
///
/// let output = conv2d(&input, &weight, None, Some((1, 1)), Some((1, 1))).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 32, 32]);
/// ```
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
        return Err(NNError::InvalidInput {
            message: format!(
                "Input channels ({}) must match weight input channels ({})",
                in_channels, weight_in_channels
            ),
        });
    }

    // Validate bias shape if provided
    if let Some(b) = bias {
        let bias_shape = b.shape().dims();
        if bias_shape != [out_channels] {
            return Err(NNError::InvalidInput {
                message: format!("Bias shape {:?} must be [{}]", bias_shape, out_channels),
            });
        }
    }

    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));

    let out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    // Calculate total output size
    let output_size = batch_size * out_channels * out_height * out_width;
    let mut output_data = vec![T::from(0.0).unwrap(); output_size];

    // Perform convolution with SIMD acceleration
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = T::from(0.0).unwrap();

                    // Convolution kernel with SIMD acceleration
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            // Bounds check with padding
                            if ih >= padding_h
                                && ih < in_height + padding_h
                                && iw >= padding_w
                                && iw < in_width + padding_w
                            {
                                let input_ih = ih - padding_h;
                                let input_iw = iw - padding_w;

                                // Extract input and weight patches for SIMD dot product
                                let mut input_patch = Vec::with_capacity(in_channels);
                                let mut weight_patch = Vec::with_capacity(in_channels);

                                for ic in 0..in_channels {
                                    let input_idx = ((b * in_channels + ic) * in_height + input_ih)
                                        * in_width
                                        + input_iw;
                                    let weight_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                        * kernel_width
                                        + kw;

                                    input_patch.push(input_data[input_idx]);
                                    weight_patch.push(weight_data[weight_idx]);
                                }

                                // conv_kernel_dot_product_simd temporarily disabled
                                // Simple dot product without SIMD
                                let mut dot_product = T::zero();
                                for (a, b) in input_patch.iter().zip(weight_patch.iter()) {
                                    dot_product = dot_product + (*a * *b);
                                }
                                sum = sum + dot_product;
                            }
                        }
                    }

                    // Add bias if provided
                    if let Some(bias_tensor) = bias {
                        sum = sum + bias_tensor.as_slice()[oc];
                    }

                    // Store result
                    let output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }

    // Apply SIMD bias addition if bias is provided
    if let Some(bias_tensor) = bias {
        // add_bias_simd temporarily disabled
        // Simple bias addition without SIMD
        for i in 0..out_channels {
            for j in 0..(out_height * out_width) {
                let idx = i * out_height * out_width + j;
                output_data[idx] = output_data[idx] + bias_tensor.as_slice()[i];
            }
        }
    }

    let output_shape = vec![batch_size, out_channels, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

/// Applies a 2D transposed convolution (deconvolution) over an input signal.
///
/// This operation is also known as deconvolution or upsampling convolution.
/// It can be used to increase the spatial dimensions of the input.
///
/// # Arguments
/// * `input` - Input tensor of shape `(N, C_in, H_in, W_in)`
/// * `weight` - Weight tensor of shape `(C_in, C_out, K_h, K_w)`
/// * `bias` - Optional bias tensor of shape `(C_out,)`
/// * `stride` - Stride for height and width `(stride_h, stride_w)`. Default: (1, 1)
/// * `padding` - Padding for height and width `(pad_h, pad_w)`. Default: (0, 0)
/// * `output_padding` - Additional padding for output `(out_pad_h, out_pad_w)`. Default: (0, 0)
///
/// # Returns
/// Output tensor of shape `(N, C_out, H_out, W_out)` where the spatial dimensions
/// are increased according to the stride and kernel parameters.
///
/// # Examples
/// ```rust
/// use nn::functional_conv::conv2d_transpose;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 16]).unwrap();
/// let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[64, 128, 3, 3]).unwrap();
///
/// let output = conv2d_transpose(&input, &weight, None, Some((2, 2)), Some((1, 1)), Some((0, 0))).unwrap();
/// // Output shape depends on stride and kernel parameters
/// ```
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
        return Err(NNError::InvalidInput {
            message: format!(
                "Input channels ({}) must match weight input channels ({})",
                in_channels, weight_in_channels
            ),
        });
    }

    let (stride_h, stride_w) = stride.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding.unwrap_or((0, 0));
    let (output_padding_h, output_padding_w) = output_padding.unwrap_or((0, 0));

    // Calculate output dimensions for transposed convolution
    let out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    // Calculate total output size
    let output_size = batch_size * out_channels * out_height * out_width;
    let mut output_data = vec![T::from(0.0).unwrap(); output_size];

    // Perform transposed convolution
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

    // Add bias if provided
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
