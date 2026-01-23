use crate::core::error::{NNError, Result};
use backend::CpuBackend;
use dtype::DataType;
use storage::DenseStorage;
use tensor::Tensor;

pub fn compute_output_length(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    (input_length + 2 * padding - kernel_size) / stride + 1
}

pub fn conv1d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    // Validate dimensions
    if weight_shape[1] != in_channels {
        return Err(NNError::ShapeMismatch {
            operation: "Conv1D".to_string(),
            expected: vec![out_channels, in_channels, kernel_size],
            actual: weight_shape.to_vec(),
        });
    }

    let output_length = compute_output_length(input_length, kernel_size, stride, padding);

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_length;
    let mut output_data = vec![T::zero(); output_size];

    // Pad input if necessary
    let padded_length = input_length + 2 * padding;
    let mut padded_input = vec![T::zero(); batch_size * in_channels * padded_length];

    if padding > 0 {
        // Copy input to padded tensor with padding
        for b in 0..batch_size {
            for c in 0..in_channels {
                for l in 0..input_length {
                    let input_idx = ((b * in_channels + c) * input_length) + l;
                    let padded_idx = ((b * in_channels + c) * padded_length) + l + padding;
                    padded_input[padded_idx] = input.as_slice()[input_idx];
                }
            }
        }
    } else {
        // No padding, just copy
        padded_input.copy_from_slice(input.as_slice());
    }

    // Perform convolution
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
                            sum = sum + padded_input[input_idx] * weight.as_slice()[weight_idx];
                        }
                    }
                }

                // Add bias if provided
                if let Some(bias_tensor) = bias {
                    sum = sum + bias_tensor.as_slice()[oc];
                }

                let output_idx = ((b * out_channels + oc) * output_length) + ol;
                output_data[output_idx] = sum;
            }
        }
    }

    Ok(Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, output_length],
    )?)
}

pub fn conv_transpose_1d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];
    let out_channels = weight_shape[1];
    let kernel_size = weight_shape[2];

    let output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_length;
    let mut output_data = vec![T::zero(); output_size];

    // Perform transposed convolution
    for b in 0..batch_size {
        for ic in 0..in_channels {
            for il in 0..input_length {
                for oc in 0..out_channels {
                    for k in 0..kernel_size {
                        // Calculate output position
                        let stride_term = il * stride;
                        let kernel_term = k;
                        let padding_term = padding;

                        // Check bounds to prevent underflow
                        if stride_term + kernel_term >= padding_term {
                            let output_pos = stride_term + kernel_term - padding_term;
                            if output_pos < output_length {
                                let input_idx = ((b * in_channels + ic) * input_length) + il;
                                let weight_idx = ((ic * out_channels + oc) * kernel_size) + k;
                                let output_idx =
                                    ((b * out_channels + oc) * output_length) + output_pos;

                                output_data[output_idx] = output_data[output_idx]
                                    + input.as_slice()[input_idx] * weight.as_slice()[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if let Some(bias_tensor) = bias {
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for ol in 0..output_length {
                    let output_idx = ((b * out_channels + oc) * output_length) + ol;
                    output_data[output_idx] = output_data[output_idx] + bias_tensor.as_slice()[oc];
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, output_length],
    )?)
}
