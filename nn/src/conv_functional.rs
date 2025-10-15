//! Functional API for convolution operations.
//!
//! This module provides functional convolution operations and utilities
//! for padding, stride calculations, and shared convolution algorithms.

use crate::error::{NNError, Result};
use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{Storage, DenseStorage, StorageFromVec};
use coeus_tensor::Tensor;

/// Pad a 2D tensor with zeros according to padding parameters.
///
/// # Arguments
/// * `input` - Input tensor of shape [batch_size, channels, height, width]
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// Padded tensor with shape [batch_size, channels, height + 2*padding_h, width + 2*padding_w]
pub fn pad_2d<B, S, T>(
    input: &Tensor<B, S, T>,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(NNError::ShapeMismatch {
            operation: "pad_2d".to_string(),
            expected: vec![0, 0, 0, 0],
            actual: input_shape.to_vec(),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let padded_height = input_height + 2 * padding_h;
    let padded_width = input_width + 2 * padding_w;

    let input_data = input.as_slice();
    let mut padded_data = vec![T::zero(); batch_size * channels * padded_height * padded_width];

    // Copy input data to padded tensor with offset
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..input_height {
                for w in 0..input_width {
                    let input_idx = ((b * channels + c) * input_height + h) * input_width + w;
                    let padded_idx = ((b * channels + c) * padded_height + (h + padding_h)) * padded_width + (w + padding_w);
                    padded_data[padded_idx] = input_data[input_idx];
                }
            }
        }
    }

    let padded_shape = [batch_size, channels, padded_height, padded_width];
    Tensor::from_vec(padded_data, &padded_shape).map_err(Into::into)
}

/// Pad a 3D tensor with zeros according to padding parameters.
///
/// # Arguments
/// * `input` - Input tensor of shape [batch_size, channels, depth, height, width]
/// * `padding_d` - Padding in depth dimension
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// Padded tensor with shape [batch_size, channels, depth + 2*padding_d, height + 2*padding_h, width + 2*padding_w]
pub fn pad_3d<B, S, T>(
    input: &Tensor<B, S, T>,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 5 {
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
    let mut padded_data = vec![T::zero(); batch_size * channels * padded_depth * padded_height * padded_width];

    // Copy input data to padded tensor with offset
    for b in 0..batch_size {
        for c in 0..channels {
            for d in 0..input_depth {
                for h in 0..input_height {
                    for w in 0..input_width {
                        let input_idx = (((b * channels + c) * input_depth + d) * input_height + h) * input_width + w;
                        let padded_idx = (((b * channels + c) * padded_depth + (d + padding_d)) * padded_height + (h + padding_h)) * padded_width + (w + padding_w);
                        padded_data[padded_idx] = input_data[input_idx];
                    }
                }
            }
        }
    }

    let padded_shape = [batch_size, channels, padded_depth, padded_height, padded_width];
    Tensor::from_vec(padded_data, &padded_shape).map_err(Into::into)
}

/// Compute output dimensions for 2D convolution.
///
/// # Arguments
/// * `input_height` - Input height
/// * `input_width` - Input width
/// * `kernel_height` - Kernel height
/// * `kernel_width` - Kernel width
/// * `stride_h` - Stride in height dimension
/// * `stride_w` - Stride in width dimension
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// (output_height, output_width)
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

/// Compute output dimensions for 3D convolution.
///
/// # Arguments
/// * `input_depth` - Input depth
/// * `input_height` - Input height
/// * `input_width` - Input width
/// * `kernel_depth` - Kernel depth
/// * `kernel_height` - Kernel height
/// * `kernel_width` - Kernel width
/// * `stride_d` - Stride in depth dimension
/// * `stride_h` - Stride in height dimension
/// * `stride_w` - Stride in width dimension
/// * `padding_d` - Padding in depth dimension
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// (output_depth, output_height, output_width)
pub fn conv3d_output_size(
    input_depth: usize,
    input_height: usize,
    input_width: usize,
    kernel_depth: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
) -> (usize, usize, usize) {
    let out_depth = (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1;
    let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
    (out_depth, out_height, out_width)
}

/// Compute output dimensions for 1D convolution.
///
/// # Arguments
/// * `input_length` - Input length
/// * `kernel_size` - Kernel size
/// * `stride` - Stride
/// * `padding` - Padding
///
/// # Returns
/// output_length
pub fn conv1d_output_size(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    (input_length + 2 * padding - kernel_size) / stride + 1
}

/// Perform 2D convolution using functional API.
/// Low-level function for custom convolution operations.
pub fn conv2d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 4 {
        return Err(NNError::InvalidInput {
            message: "Input must be 4D [batch, channels, height, width]".to_string(),
        });
    }

    if weight_shape.len() != 4 {
        return Err(NNError::InvalidInput {
            message: "Weight must be 4D [out_channels, in_channels, kernel_h, kernel_w]".to_string(),
        });
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if weight_shape[1] != in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "conv2d".to_string(),
            expected: vec![out_channels, in_channels, kernel_height, kernel_width],
            actual: weight_shape.to_vec(),
        });
    }

    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;

    let (output_height, output_width) = conv2d_output_size(
        input_height, input_width, kernel_height, kernel_width,
        stride_h, stride_w, padding_h, padding_w
    );

    // Pad input if necessary
    let padded_input = if padding_h > 0 || padding_w > 0 {
        pad_2d(input, padding_h, padding_w)?
    } else {
        input.clone()
    };

    let padded_shape = padded_input.shape().dims();
    let padded_height = padded_shape[2];
    let padded_width = padded_shape[3];

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_height * output_width;
    let mut output_data = vec![T::zero(); output_size];

    let input_data = padded_input.as_slice();
    let weight_data = weight.as_slice();

    // Perform convolution
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = T::zero();

                    // Convolve over input channels and kernel
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                if ih < padded_height && iw < padded_width {
                                    let input_idx = ((b * in_channels + ic) * padded_height + ih) * padded_width + iw;
                                    let weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    sum = sum + input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }

                    // Add bias if provided
                    if let Some(bias_tensor) = bias {
                        let bias_data = bias_tensor.as_slice();
                        sum = sum + bias_data[oc];
                    }

                    let output_idx = ((b * out_channels + oc) * output_height + oh) * output_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }

    let output_shape = [batch_size, out_channels, output_height, output_width];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

/// Perform 3D convolution using functional API.
///
/// # Arguments
/// * `input` - Input tensor [batch_size, in_channels, depth, height, width]
/// * `weight` - Weight tensor [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
/// * `bias` - Optional bias tensor [out_channels]
/// * `stride` - (stride_d, stride_h, stride_w)
/// * `padding` - (padding_d, padding_h, padding_w)
///
/// # Returns
/// Output tensor [batch_size, out_channels, output_depth, output_height, output_width]
pub fn conv3d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: "Input must be 5D [batch, channels, depth, height, width]".to_string(),
        });
    }

    if weight_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: "Weight must be 5D [out_channels, in_channels, kernel_d, kernel_h, kernel_w]".to_string(),
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
            expected: vec![out_channels, in_channels, kernel_depth, kernel_height, kernel_width],
            actual: weight_shape.to_vec(),
        });
    }

    let (stride_d, stride_h, stride_w) = stride;
    let (padding_d, padding_h, padding_w) = padding;

    let (output_depth, output_height, output_width) = conv3d_output_size(
        input_depth, input_height, input_width, kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w, padding_d, padding_h, padding_w
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

                                        if id < padded_depth && ih < padded_height && iw < padded_width {
                                            let input_idx = (((b * in_channels + ic) * padded_depth + id) * padded_height + ih) * padded_width + iw;
                                            let weight_idx = (((oc * in_channels + ic) * kernel_depth + kd) * kernel_height + kh) * kernel_width + kw;
                                            sum = sum + input_data[input_idx] * weight_data[weight_idx];
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

                        let output_idx = (((b * out_channels + oc) * output_depth + od) * output_height + oh) * output_width + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }

    let output_shape = [batch_size, out_channels, output_depth, output_height, output_width];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

/// Perform 1D convolution using functional API.
///
/// # Arguments
/// * `input` - Input tensor [batch_size, in_channels, length]
/// * `weight` - Weight tensor [out_channels, in_channels, kernel_size]
/// * `bias` - Optional bias tensor [out_channels]
/// * `stride` - Stride
/// * `padding` - Padding
///
/// # Returns
/// Output tensor [batch_size, out_channels, output_length]
pub fn conv1d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: "Input must be 3D [batch, channels, length]".to_string(),
        });
    }

    if weight_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: "Weight must be 3D [out_channels, in_channels, kernel_size]".to_string(),
        });
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    if weight_shape[1] != in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "conv1d".to_string(),
            expected: vec![out_channels, in_channels, kernel_size],
            actual: weight_shape.to_vec(),
        });
    }

    let output_length = conv1d_output_size(input_length, kernel_size, stride, padding);

    // Pad input if necessary
    let padded_input = if padding > 0 {
        let padded_length = input_length + 2 * padding;
        let mut padded_data = vec![T::zero(); batch_size * in_channels * padded_length];

        for b in 0..batch_size {
            for c in 0..in_channels {
                for l in 0..input_length {
                    let input_idx = ((b * in_channels + c) * input_length) + l;
                    let padded_idx = ((b * in_channels + c) * padded_length) + l + padding;
                    padded_data[padded_idx] = input.as_slice()[input_idx];
                }
            }
        }
        Tensor::from_vec(padded_data, &[batch_size, in_channels, padded_length])?
    } else {
        input.clone()
    };

    let padded_shape = padded_input.shape().dims();
    let padded_length = padded_shape[2];

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_length;
    let mut output_data = vec![T::zero(); output_size];

    let input_data = padded_input.as_slice();
    let weight_data = weight.as_slice();

    // Perform convolution
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..output_length {
                let mut sum = T::zero();

                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        let input_pos = ol * stride + k;
                        if input_pos < padded_length {
                            let input_idx = ((b * in_channels + ic) * padded_length) + input_pos;
                            let weight_idx = ((oc * in_channels + ic) * kernel_size) + k;
                            sum = sum + input_data[input_idx] * weight_data[weight_idx];
                        }
                    }
                }

                // Add bias if provided
                if let Some(bias_tensor) = bias {
                    let bias_data = bias_tensor.as_slice();
                    sum = sum + bias_data[oc];
                }

                let output_idx = ((b * out_channels + oc) * output_length) + ol;
                output_data[output_idx] = sum;
            }
        }
    }

    let output_shape = [batch_size, out_channels, output_length];
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    type TestTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_pad_2d() {
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 24], &[1, 1, 4, 6]).unwrap(); // 1x1x4x6
        let padded = pad_2d(&input, 1, 2).unwrap();
        assert_eq!(padded.shape().dims(), &[1, 1, 6, 10]); // padding 1,2 -> +2, +4
    }

    #[test]
    fn test_pad_3d() {
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 24], &[1, 1, 2, 3, 4]).unwrap(); // 1x1x2x3x4
        let padded = pad_3d(&input, 1, 1, 1).unwrap();
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 5, 6]); // padding 1,1,1 -> +2,+2,+2
    }

    #[test]
    fn test_conv1d_functional() {
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 15], &[1, 1, 15]).unwrap();
        let weight = TestTensor::from_vec(vec![Float32::new(0.5); 12], &[2, 1, 6]).unwrap();
        let bias = TestTensor::from_vec(vec![Float32::new(0.1); 2], &[2]).unwrap();
        let output = conv1d(&input, &weight, Some(&bias), 1, 0).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 10]);
    }

    #[test]
    fn test_conv2d_functional() {
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 24], &[1, 1, 4, 6]).unwrap();
        let weight = TestTensor::from_vec(vec![Float32::new(0.5); 18], &[2, 1, 3, 3]).unwrap();
        let bias = TestTensor::from_vec(vec![Float32::new(0.1); 2], &[2]).unwrap();
        let output = conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0)).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 2, 4]);
    }

    #[test]
    fn test_conv3d_functional() {
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 60], &[1, 1, 3, 4, 5]).unwrap();
        let weight = TestTensor::from_vec(vec![Float32::new(0.5); 54], &[2, 1, 3, 3, 3]).unwrap();
        let bias = TestTensor::from_vec(vec![Float32::new(0.1); 2], &[2]).unwrap();
        let output = conv3d(&input, &weight, Some(&bias), (1, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 1, 2, 3]);
    }

    #[test]
    fn test_output_size_calculations() {
        assert_eq!(conv1d_output_size(10, 3, 1, 0), 8);
        assert_eq!(conv2d_output_size(10, 10, 3, 3, 1, 1, 0, 0), (8, 8));
        assert_eq!(conv3d_output_size(5, 10, 10, 3, 3, 3, 1, 1, 1, 0, 0, 0), (3, 8, 8));
    }
}
