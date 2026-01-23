use crate::core::error::Result;
use backend::CpuBackend;
use dtype::DataType;
use storage::DenseStorage;
use tensor::Tensor;

pub fn conv2d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Calculate output dimensions
    let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

    // Pad input if necessary
    let padded_input = if padding_h > 0 || padding_w > 0 {
        let padded_height = input_height + 2 * padding_h;
        let padded_width = input_width + 2 * padding_w;
        let mut padded_data =
            vec![T::zero(); batch_size * in_channels * padded_height * padded_width];

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..input_height {
                    for w in 0..input_width {
                        let input_idx =
                            ((b * in_channels + c) * input_height + h) * input_width + w;
                        let padded_idx = ((b * in_channels + c) * padded_height + (h + padding_h))
                            * padded_width
                            + (w + padding_w);
                        padded_data[padded_idx] = input.as_slice()[input_idx];
                    }
                }
            }
        }
        Tensor::from_vec(
            padded_data,
            &[batch_size, in_channels, padded_height, padded_width],
        )?
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

                    // Convolve over input channels, kernel height, kernel width
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                // Input position (accounting for stride)
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Input data index
                                let input_idx = ((b * in_channels + ic) * padded_height + ih)
                                    * padded_width
                                    + iw;
                                let input_val = input_data[input_idx];

                                // Weight data index
                                let weight_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                    * kernel_width
                                    + kw;
                                let weight_val = weight_data[weight_idx];

                                sum = sum + input_val * weight_val;
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_tensor) = bias {
                        let bias_data = bias_tensor.as_slice();
                        sum = sum + bias_data[oc];
                    }

                    // Output data index
                    let output_idx =
                        ((b * out_channels + oc) * output_height + oh) * output_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }

    let output_shape = [batch_size, out_channels, output_height, output_width];
    Ok(Tensor::from_vec(output_data, &output_shape)?)
}

pub fn conv_transpose_2d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_h: usize,
    output_padding_w: usize,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    let out_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Calculate output dimensions
    let output_height =
        (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    let output_width =
        (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_height * output_width;
    let mut output_data = vec![T::zero(); output_size];

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    // Perform transposed convolution
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        for ic in 0..in_channels {
            for ih in 0..input_height {
                for iw in 0..input_width {
                    for oc in 0..out_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                // Calculate output position
                                let stride_term_h = ih * stride_h;
                                let stride_term_w = iw * stride_w;

                                // Check bounds
                                if stride_term_h + kh >= padding_h
                                    && stride_term_w + kw >= padding_w
                                {
                                    let oh = stride_term_h + kh - padding_h;
                                    let ow = stride_term_w + kw - padding_w;

                                    if oh < output_height && ow < output_width {
                                        let input_idx = ((b * in_channels + ic) * input_height
                                            + ih)
                                            * input_width
                                            + iw;
                                        let weight_idx = ((ic * out_channels + oc) * kernel_height
                                            + kh)
                                            * kernel_width
                                            + kw;
                                        let output_idx = ((b * out_channels + oc) * output_height
                                            + oh)
                                            * output_width
                                            + ow;

                                        output_data[output_idx] = output_data[output_idx]
                                            + input_data[input_idx] * weight_data[weight_idx];
                                    }
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
            for (oc, &bias) in bias_data.iter().enumerate().take(out_channels) {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let output_idx =
                            ((b * out_channels + oc) * output_height + oh) * output_width + ow;
                        output_data[output_idx] = output_data[output_idx] + bias;
                    }
                }
            }
        }
    }

    let output_shape = [batch_size, out_channels, output_height, output_width];
    Ok(Tensor::from_vec(output_data, &output_shape)?)
}
