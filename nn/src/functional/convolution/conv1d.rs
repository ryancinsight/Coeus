//! 1D Convolution operations for neural networks.

use crate::core::error::{NNError, Result};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec};
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
pub fn conv1d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 3usize {
        return Err(NNError::InvalidInput {
            message: "Input must be 3D [batch, channels, length]".to_string(),
        });
    }

    if weight_shape.len() != 3usize {
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
