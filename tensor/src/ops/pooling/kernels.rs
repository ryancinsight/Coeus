use crate::Tensor;
use crate::error::TensorError;
use crate::Result;
use backend::CpuBackend;
use dtype::DataType;
use storage::DenseStorage;
use num_traits::Float;

pub fn max_pool1d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 3 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 3D input (N, C, L), got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_length = input_shape[2];

    let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
    
    let output_length = if ceil_mode {
        (input_length + 2 * padding - dilated_kernel_size + stride - 1) / stride + 1
    } else {
        (input_length + 2 * padding - dilated_kernel_size) / stride + 1
    };

    // Correct for corner cases where ceil_mode might give an output larger than valid starts
    // PyTorch ensures the last window starts within the input (including padding).
    // But simple formula might overshoot.
    // We will stick to the basic calculation for now and clamp bounds in the loop if needed.
    // Actually, PyTorch checks: if (output_length - 1) * stride >= input_length + padding, it might reduce output_size?
    // We'll stick to the formula: L_out = ceil(...) vs floor(...)

    // Ensure output_length is at least 1? Or 0 if input is too small?
    if input_length + 2 * padding < dilated_kernel_size {
         return Err(TensorError::InvalidInput {
            message: format!("Input too small for pooling: length={}, padding={}, kernel={}, dilation={}", input_length, padding, kernel_size, dilation),
        });
    }

    let output_shape = vec![batch_size, channels, output_length];
    let mut output_data = Vec::with_capacity(batch_size * channels * output_length);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_length {
                let mut max_val = T::from(f64::NEG_INFINITY).unwrap();
                let mut set = false;

                for k in 0..kernel_size {
                    let input_pos = (ol * stride + k * dilation) as isize - padding as isize;

                    if input_pos >= 0 && input_pos < input_length as isize {
                        let idx = b * (channels * input_length)
                            + c * input_length
                            + input_pos as usize;
                        let val = input_data[idx];
                        if !set || val > max_val {
                            max_val = val;
                            set = true;
                        }
                    }
                }
                // If the window was completely out of bounds (should not happen with correct output_length),
                // we might have -inf. But valid pooling windows usually overlap some valid input.
                // If padding is large, it might be that all sampled points are padding.
                // PyTorch uses -inf for padding (implicitly ignores it for MaxPool).
                // So keeping default max_val is correct.
                
                output_data.push(max_val);
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

pub fn max_pool2d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 4D input (N, C, H, W), got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    let (k_h, k_w) = kernel_size;
    let (s_h, s_w) = stride;
    let (p_h, p_w) = padding;
    let (d_h, d_w) = dilation;

    let dilated_k_h = (k_h - 1) * d_h + 1;
    let dilated_k_w = (k_w - 1) * d_w + 1;

    let output_h = if ceil_mode {
        (input_h + 2 * p_h - dilated_k_h + s_h - 1) / s_h + 1
    } else {
        (input_h + 2 * p_h - dilated_k_h) / s_h + 1
    };
    
    let output_w = if ceil_mode {
        (input_w + 2 * p_w - dilated_k_w + s_w - 1) / s_w + 1
    } else {
        (input_w + 2 * p_w - dilated_k_w) / s_w + 1
    };

    if input_h + 2 * p_h < dilated_k_h || input_w + 2 * p_w < dilated_k_w {
         return Err(TensorError::InvalidInput {
            message: format!("Input too small for pooling 2d"),
        });
    }

    let output_shape = vec![batch_size, channels, output_h, output_w];
    let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let mut max_val = T::from(f64::NEG_INFINITY).unwrap();
                    let mut set = false;

                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let h_in = (oh * s_h + kh * d_h) as isize - p_h as isize;
                            let w_in = (ow * s_w + kw * d_w) as isize - p_w as isize;

                            if h_in >= 0 && h_in < input_h as isize && w_in >= 0 && w_in < input_w as isize {
                                let idx = b * (channels * input_h * input_w)
                                    + c * (input_h * input_w)
                                    + (h_in as usize) * input_w
                                    + (w_in as usize);
                                let val = input_data[idx];
                                if !set || val > max_val {
                                    max_val = val;
                                    set = true;
                                }
                            }
                        }
                    }
                    output_data.push(max_val);
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

pub fn max_pool3d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    ceil_mode: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 5 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 5D input (N, C, D, H, W), got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_d = input_shape[2];
    let input_h = input_shape[3];
    let input_w = input_shape[4];

    let (k_d, k_h, k_w) = kernel_size;
    let (s_d, s_h, s_w) = stride;
    let (p_d, p_h, p_w) = padding;
    let (d_dep, d_h, d_w) = dilation;

    let dilated_k_d = (k_d - 1) * d_dep + 1;
    let dilated_k_h = (k_h - 1) * d_h + 1;
    let dilated_k_w = (k_w - 1) * d_w + 1;

    let output_d = if ceil_mode { (input_d + 2 * p_d - dilated_k_d + s_d - 1) / s_d + 1 } else { (input_d + 2 * p_d - dilated_k_d) / s_d + 1 };
    let output_h = if ceil_mode { (input_h + 2 * p_h - dilated_k_h + s_h - 1) / s_h + 1 } else { (input_h + 2 * p_h - dilated_k_h) / s_h + 1 };
    let output_w = if ceil_mode { (input_w + 2 * p_w - dilated_k_w + s_w - 1) / s_w + 1 } else { (input_w + 2 * p_w - dilated_k_w) / s_w + 1 };

    if input_d + 2 * p_d < dilated_k_d || input_h + 2 * p_h < dilated_k_h || input_w + 2 * p_w < dilated_k_w {
         return Err(TensorError::InvalidInput {
            message: format!("Input too small for pooling 3d"),
        });
    }

    let output_shape = vec![batch_size, channels, output_d, output_h, output_w];
    let mut output_data = Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_d {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        let mut max_val = T::from(f64::NEG_INFINITY).unwrap();
                        let mut set = false;

                        for kd in 0..k_d {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let d_in = (od * s_d + kd * d_dep) as isize - p_d as isize;
                                    let h_in = (oh * s_h + kh * d_h) as isize - p_h as isize;
                                    let w_in = (ow * s_w + kw * d_w) as isize - p_w as isize;

                                    if d_in >= 0 && d_in < input_d as isize &&
                                       h_in >= 0 && h_in < input_h as isize &&
                                       w_in >= 0 && w_in < input_w as isize {
                                        let idx = b * (channels * input_d * input_h * input_w)
                                            + c * (input_d * input_h * input_w)
                                            + (d_in as usize) * (input_h * input_w)
                                            + (h_in as usize) * input_w
                                            + (w_in as usize);
                                        let val = input_data[idx];
                                        if !set || val > max_val {
                                            max_val = val;
                                            set = true;
                                        }
                                    }
                                }
                            }
                        }
                        output_data.push(max_val);
                    }
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

pub fn avg_pool1d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 3 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 3D input (batch, channels, length), got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_length = input_shape[2];

    let output_length = if ceil_mode {
        (input_length + 2 * padding - kernel_size + stride - 1) / stride + 1
    } else {
        (input_length + 2 * padding - kernel_size) / stride + 1
    };

    let output_shape = vec![batch_size, channels, output_length];
    let mut output_data = Vec::with_capacity(batch_size * channels * output_length);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_length {
                let mut sum = T::zero();
                let mut count = 0;

                for k in 0..kernel_size {
                    let input_pos = (ol * stride + k) as isize - padding as isize;

                    if input_pos >= 0 && input_pos < input_length as isize {
                        let idx = b * (channels * input_length) + c * input_length + input_pos as usize;
                        sum = sum + input_data[idx];
                        count += 1;
                    }
                }

                let val = if count > 0 {
                    let divisor = if count_include_pad { kernel_size } else { count };
                    sum / T::from(divisor as f64).unwrap()
                } else {
                    T::zero()
                };
                output_data.push(val);
            }
        }
    }
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

pub fn avg_pool2d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 4D input [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    let h_out = if ceil_mode {
        (input_h + 2 * padding.0 - kernel_size.0 + stride.0 - 1) / stride.0 + 1
    } else {
        (input_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1
    };
    let w_out = if ceil_mode {
        (input_w + 2 * padding.1 - kernel_size.1 + stride.1 - 1) / stride.1 + 1
    } else {
        (input_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1
    };

    let output_shape = vec![batch_size, channels, h_out, w_out];
    let mut output_data = Vec::with_capacity(batch_size * channels * h_out * w_out);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = T::zero();
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                             let h_in = (oh * stride.0 + kh) as isize - padding.0 as isize;
                             let w_in = (ow * stride.1 + kw) as isize - padding.1 as isize;

                             if h_in >= 0 && h_in < input_h as isize && w_in >= 0 && w_in < input_w as isize {
                                 let idx = ((b * channels + c) * input_h + h_in as usize) * input_w + w_in as usize;
                                 sum = sum + input_data[idx];
                                 count += 1;
                             }
                        }
                    }

                    let val = if count > 0 {
                          let divisor = if count_include_pad { kernel_size.0 * kernel_size.1 } else { count };
                          sum / T::from(divisor as f64).unwrap()
                    } else {
                        T::zero()
                    };
                    output_data.push(val);
                }
            }
        }
    }
     Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}

pub fn avg_pool3d_cpu_dense<T>(
    input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>>
where
    T: DataType + num_traits::Float + num_traits::FromPrimitive + PartialOrd + std::fmt::Debug,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 5 {
        return Err(TensorError::InvalidInput {
            message: format!("Expected 5D input [N, C, D, H, W], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_d = input_shape[2];
    let input_h = input_shape[3];
    let input_w = input_shape[4];

    let d_out = if ceil_mode {
        (input_d + 2 * padding.0 - kernel_size.0 + stride.0 - 1) / stride.0 + 1
    } else {
        (input_d + 2 * padding.0 - kernel_size.0) / stride.0 + 1
    };
     let h_out = if ceil_mode {
        (input_h + 2 * padding.1 - kernel_size.1 + stride.1 - 1) / stride.1 + 1
    } else {
        (input_h + 2 * padding.1 - kernel_size.1) / stride.1 + 1
    };
    let w_out = if ceil_mode {
        (input_w + 2 * padding.2 - kernel_size.2 + stride.2 - 1) / stride.2 + 1
    } else {
        (input_w + 2 * padding.2 - kernel_size.2) / stride.2 + 1
    };

    let output_shape = vec![batch_size, channels, d_out, h_out, w_out];
    let mut output_data = Vec::with_capacity(batch_size * channels * d_out * h_out * w_out);
    let input_data = input.as_slice();

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..d_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = T::zero();
                        let mut count = 0;

                         for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let d_in = (od * stride.0 + kd) as isize - padding.0 as isize;
                                    let h_in = (oh * stride.1 + kh) as isize - padding.1 as isize;
                                    let w_in = (ow * stride.2 + kw) as isize - padding.2 as isize;

                                    if d_in >= 0 && d_in < input_d as isize &&
                                       h_in >= 0 && h_in < input_h as isize &&
                                       w_in >= 0 && w_in < input_w as isize {
                                        let idx = ((b * channels + c) * input_d * input_h * input_w)
                                            + (d_in as usize) * input_h * input_w
                                            + (h_in as usize) * input_w
                                            + (w_in as usize);
                                        sum = sum + input_data[idx];
                                        count += 1;
                                    }
                                }
                            }
                         }

                        let val = if count > 0 {
                             let divisor = if count_include_pad { kernel_size.0 * kernel_size.1 * kernel_size.2 } else { count };
                             sum / T::from(divisor as f64).unwrap()
                        } else {
                            T::zero()
                        };
                        output_data.push(val);
                    }
                }
            }
        }
    }
    Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
}
