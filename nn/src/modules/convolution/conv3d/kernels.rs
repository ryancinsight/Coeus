use crate::core::error::Result;
use backend::CpuBackend;
use dtype::DataType;
use storage::DenseStorage;
use tensor::Tensor;

pub fn conv3d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
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
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];
    let out_channels = weight_shape[0];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    // Calculate output dimensions
    let output_depth = (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1;
    let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

    // Pad input if necessary
    let padded_input = if padding_d > 0 || padding_h > 0 || padding_w > 0 {
        let padded_depth = input_depth + 2 * padding_d;
        let padded_height = input_height + 2 * padding_h;
        let padded_width = input_width + 2 * padding_w;
        let mut padded_data =
            vec![T::zero(); batch_size * in_channels * padded_depth * padded_height * padded_width];

        for b in 0..batch_size {
            for c in 0..in_channels {
                for d in 0..input_depth {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            let input_idx =
                                (((b * in_channels + c) * input_depth + d) * input_height + h)
                                    * input_width
                                    + w;
                            let padded_idx = (((b * in_channels + c) * padded_depth
                                + (d + padding_d))
                                * padded_height
                                + (h + padding_h))
                                * padded_width
                                + (w + padding_w);
                            padded_data[padded_idx] = input.as_slice()[input_idx];
                        }
                    }
                }
            }
        }
        Tensor::from_vec(
            padded_data,
            &[
                batch_size,
                in_channels,
                padded_depth,
                padded_height,
                padded_width,
            ],
        )?
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

                        // Add bias if present
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
    Ok(Tensor::from_vec(output_data, &output_shape)?)
}

pub fn conv_transpose_3d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let (stride_d, stride_h, stride_w) = stride;
    let (padding_d, padding_h, padding_w) = padding;
    let (output_padding_d, output_padding_h, output_padding_w) = output_padding;

    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];
    let out_channels = weight_shape[1];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    // Calculate output dimensions
    let output_depth =
        (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    let output_height =
        (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    let output_width =
        (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_depth * output_height * output_width;
    let mut output_data = vec![T::zero(); output_size];

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    // Perform transposed convolution
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        for ic in 0..in_channels {
            for id in 0..input_depth {
                for ih in 0..input_height {
                    for iw in 0..input_width {
                        for oc in 0..out_channels {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let stride_term_d = id * stride_d;
                                        let stride_term_h = ih * stride_h;
                                        let stride_term_w = iw * stride_w;

                                        if stride_term_d + kd >= padding_d
                                            && stride_term_h + kh >= padding_h
                                            && stride_term_w + kw >= padding_w
                                        {
                                            let od = stride_term_d + kd - padding_d;
                                            let oh = stride_term_h + kh - padding_h;
                                            let ow = stride_term_w + kw - padding_w;

                                            if od < output_depth
                                                && oh < output_height
                                                && ow < output_width
                                            {
                                                let input_idx =
                                                    (((b * in_channels + ic) * input_depth + id)
                                                        * input_height
                                                        + ih)
                                                        * input_width
                                                        + iw;
                                                let weight_idx = (((ic * out_channels + oc)
                                                    * kernel_depth
                                                    + kd)
                                                    * kernel_height
                                                    + kh)
                                                    * kernel_width
                                                    + kw;
                                                let output_idx =
                                                    (((b * out_channels + oc) * output_depth + od)
                                                        * output_height
                                                        + oh)
                                                        * output_width
                                                        + ow;

                                                output_data[output_idx] = output_data[output_idx]
                                                    + input_data[input_idx]
                                                        * weight_data[weight_idx];
                                            }
                                        }
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
                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let output_idx = (((b * out_channels + oc) * output_depth + od)
                                * output_height
                                + oh)
                                * output_width
                                + ow;
                            output_data[output_idx] = output_data[output_idx] + bias;
                        }
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
    Ok(Tensor::from_vec(output_data, &output_shape)?)
}
