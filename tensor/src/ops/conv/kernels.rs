use crate::Result;
use crate::{Tensor, TensorError};
use backend::CpuBackend;
use dtype::DataType;
use storage::DenseStorage;

// =========================================================================================
// Helper Functions
// =========================================================================================

pub fn compute_output_length(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    (input_length + 2 * padding - kernel_size) / stride + 1
}

// =========================================================================================
// Conv1d Kernels
// =========================================================================================

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
        return Err(TensorError::ShapeMismatch {
            expected: vec![out_channels, in_channels, kernel_size],
            actual: weight_shape.to_vec(),
            operation: "conv1d weights verification",
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

    Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, output_length],
    )
}

pub fn conv_transpose_1d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    dilation: usize,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];
    let out_channels_per_group = weight_shape[1];
    let out_channels = out_channels_per_group * groups;
    let kernel_size = weight_shape[2];

    if in_channels % groups != 0 {
        return Err(TensorError::InvalidInput {
            message: format!("Input channels {} must be divisible by groups {}", in_channels, groups),
        });
    }

    if weight_shape[0] != in_channels {
         return Err(TensorError::ShapeMismatch {
            expected: vec![in_channels, out_channels_per_group, kernel_size],
            actual: weight_shape.to_vec(),
            operation: "conv_transpose1d weights verification",
        });
    }

    // Effective kernel size with dilation
    let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
    let output_length = (input_length - 1) * stride - 2 * padding + dilated_kernel_size + output_padding;

    // Initialize output tensor
    let output_size = batch_size * out_channels * output_length;
    let mut output_data = vec![T::zero(); output_size];

    let in_channels_per_group = in_channels / groups;

    let input_data = input.as_slice();
    let weight_data = weight.as_slice();

    // Perform transposed convolution
    for b in 0..batch_size {
        for g in 0..groups {
            let ic_start = g * in_channels_per_group;
            let oc_start = g * out_channels_per_group;

            for ic_off in 0..in_channels_per_group {
                let ic = ic_start + ic_off;
                for il in 0..input_length {
                    for oc_off in 0..out_channels_per_group {
                        let oc = oc_start + oc_off;
                        for k in 0..kernel_size {
                            // Calculate output position taking dilation into account
                            let stride_term = il * stride;
                            let kernel_term = k * dilation;
                            let padding_term = padding;

                            // Check bounds to prevent underflow
                            if stride_term + kernel_term >= padding_term {
                                let output_pos = stride_term + kernel_term - padding_term;
                                if output_pos < output_length {
                                    let input_idx = ((b * in_channels + ic) * input_length) + il;
                                    // Weight shape: [in_channels, out_channels_per_group, kernel_size]
                                    let weight_idx = ((ic * out_channels_per_group + oc_off) * kernel_size) + k;
                                    let output_idx =
                                        ((b * out_channels + oc) * output_length) + output_pos;

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

    // Add bias if provided
    if let Some(bias_tensor) = bias {
        let bias_data = bias_tensor.as_slice();
        if bias_data.len() != out_channels {
             return Err(TensorError::ShapeMismatch {
                expected: vec![out_channels],
                actual: vec![bias_data.len()],
                operation: "conv_transpose1d bias verification",
            });
        }
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for ol in 0..output_length {
                    let output_idx = ((b * out_channels + oc) * output_length) + ol;
                    output_data[output_idx] = output_data[output_idx] + bias_data[oc];
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, output_length],
    )
}

// =========================================================================================
// Conv2d Kernels
// =========================================================================================

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
    Tensor::from_vec(output_data, &output_shape)
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
    Tensor::from_vec(output_data, &output_shape)
}

// =========================================================================================
// Conv3d Kernels
// =========================================================================================

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
    Tensor::from_vec(output_data, &output_shape)
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
    Tensor::from_vec(output_data, &output_shape)
}
