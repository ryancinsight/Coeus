//! Pooling functions for neural networks.
//!
//! This module provides stateless pooling operations for spatial downsampling
//! of feature maps in convolutional neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};

/// Applies 2D max pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape `(N, C, H_in, W_in)`
/// * `kernel_size` - Size of the pooling window `(kh, kw)`
/// * `stride` - Stride of the pooling window `(sh, sw)`. If None, defaults to kernel_size
/// * `padding` - Padding added to both sides of the input `(ph, pw)`
///
/// # Returns
/// Output tensor of shape `(N, C, H_out, W_out)` where:
/// - `H_out = floor((H_in + 2*ph - kh) / sh + 1)`
/// - `W_out = floor((W_in + 2*pw - kw) / sw + 1)`
///
/// # Examples
/// ```rust
/// use nn::functional_pooling::max_pool2d;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32]).unwrap();
/// let output = max_pool2d(&input, (2, 2), Some((2, 2)), (0, 0)).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 16, 16]);
/// ```
pub fn max_pool2d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    let stride = stride.unwrap_or(kernel_size);
    if stride.0 == 0 || stride.1 == 0 {
        return Err(NNError::InvalidInput {
            message: "Stride must be > 0".to_string(),
        });
    }

    let padded_h = input_h
        .checked_add(
            padding
                .0
                .checked_mul(2)
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Padding overflow".to_string(),
                })?,
        )
        .ok_or_else(|| NNError::InvalidInput {
            message: "Input height overflow".to_string(),
        })?;
    let padded_w = input_w
        .checked_add(
            padding
                .1
                .checked_mul(2)
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Padding overflow".to_string(),
                })?,
        )
        .ok_or_else(|| NNError::InvalidInput {
            message: "Input width overflow".to_string(),
        })?;

    if padded_h < kernel_size.0 || padded_w < kernel_size.1 {
        return Err(NNError::InvalidInput {
            message: "Kernel size exceeds padded input".to_string(),
        });
    }

    let output_h = (padded_h - kernel_size.0) / stride.0 + 1;
    let output_w = (padded_w - kernel_size.1) / stride.1 + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let mut max_val = T::neg_infinity();

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let h_in = oh * stride.0 + kh;
                            let w_in = ow * stride.1 + kw;

                            if h_in >= padding.0
                                && h_in < input_h + padding.0
                                && w_in >= padding.1
                                && w_in < input_w + padding.1
                            {
                                let h_actual = h_in - padding.0;
                                let w_actual = w_in - padding.1;

                                if h_actual < input_h && w_actual < input_w {
                                    let idx = ((n * channels + c) * input_h + h_actual) * input_w
                                        + w_actual;
                                    let val = input_data[idx];
                                    if val > max_val {
                                        max_val = val;
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

    let output_shape = vec![batch_size, channels, output_h, output_w];
    Ok(Tensor::from_vec_with_backend(
        output_data,
        &output_shape,
        input.backend().clone(),
    )?)
}

/// Applies 2D average pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape `(N, C, H_in, W_in)`
/// * `kernel_size` - Size of the pooling window `(kh, kw)`
/// * `stride` - Stride of the pooling window `(sh, sw)`. If None, defaults to kernel_size
/// * `padding` - Padding added to both sides of the input `(ph, pw)`
///
/// # Returns
/// Output tensor of shape `(N, C, H_out, W_out)` where the output is the average
/// of each pooling window.
///
/// # Examples
/// ```rust
/// use nn::functional_pooling::avg_pool2d;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32]).unwrap();
/// let output = avg_pool2d(&input, (2, 2), Some((2, 2)), (0, 0)).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 16, 16]);
/// ```
pub fn avg_pool2d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 4usize {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    let stride = stride.unwrap_or(kernel_size);
    if stride.0 == 0 || stride.1 == 0 {
        return Err(NNError::InvalidInput {
            message: "Stride must be > 0".to_string(),
        });
    }

    let padded_h = input_h
        .checked_add(
            padding
                .0
                .checked_mul(2)
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Padding overflow".to_string(),
                })?,
        )
        .ok_or_else(|| NNError::InvalidInput {
            message: "Input height overflow".to_string(),
        })?;
    let padded_w = input_w
        .checked_add(
            padding
                .1
                .checked_mul(2)
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Padding overflow".to_string(),
                })?,
        )
        .ok_or_else(|| NNError::InvalidInput {
            message: "Input width overflow".to_string(),
        })?;

    if padded_h < kernel_size.0 || padded_w < kernel_size.1 {
        return Err(NNError::InvalidInput {
            message: "Kernel size exceeds padded input".to_string(),
        });
    }

    let output_h = (padded_h - kernel_size.0) / stride.0 + 1;
    let output_w = (padded_w - kernel_size.1) / stride.1 + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let mut sum = T::zero();
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let h_in = oh * stride.0 + kh;
                            let w_in = ow * stride.1 + kw;

                            if h_in >= padding.0
                                && h_in < input_h + padding.0
                                && w_in >= padding.1
                                && w_in < input_w + padding.1
                            {
                                let h_actual = h_in - padding.0;
                                let w_actual = w_in - padding.1;

                                if h_actual < input_h && w_actual < input_w {
                                    let idx = ((n * channels + c) * input_h + h_actual) * input_w
                                        + w_actual;
                                    sum = sum + input_data[idx];
                                    count += 1;
                                }
                            }
                        }
                    }

                    let avg = if count > 0 {
                        sum / T::from(count).ok_or_else(|| NNError::InvalidInput {
                            message: "Pooling window element count not representable".to_string(),
                        })?
                    } else {
                        T::zero()
                    };
                    output_data.push(avg);
                }
            }
        }
    }

    let output_shape = vec![batch_size, channels, output_h, output_w];
    Ok(Tensor::from_vec_with_backend(
        output_data,
        &output_shape,
        input.backend().clone(),
    )?)
}

/// Applies 1D max pooling over an input signal.
pub fn max_pool1d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 3D [N, C, L], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_l = input_shape[2];

    let stride = stride.unwrap_or(kernel_size);
    let output_l = (input_l + 2 * padding - kernel_size) / stride + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_l);

    for n in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_l {
                let mut max_val = T::neg_infinity();
                for kl in 0..kernel_size {
                    let l_in = ol * stride + kl;
                    if l_in >= padding && l_in < input_l + padding {
                        let l_actual = l_in - padding;
                        let idx = (n * channels + c) * input_l + l_actual;
                        let val = input_data[idx];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                output_data.push(max_val);
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        &[batch_size, channels, output_l],
        input.backend().clone(),
    )?)
}

/// Applies 1D average pooling over an input signal.
pub fn avg_pool1d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 3D [N, C, L], got {}D", input_shape.len()),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_l = input_shape[2];

    let stride = stride.unwrap_or(kernel_size);
    let output_l = (input_l + 2 * padding - kernel_size) / stride + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_l);

    for n in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_l {
                let mut sum = T::zero();
                let mut count = 0;
                for kl in 0..kernel_size {
                    let l_in = ol * stride + kl;
                    if l_in >= padding && l_in < input_l + padding {
                        let l_actual = l_in - padding;
                        let idx = (n * channels + c) * input_l + l_actual;
                        sum = sum + input_data[idx];
                        count += 1;
                    }
                }
                let avg = if count > 0 {
                    sum / T::from(count).unwrap()
                } else {
                    T::zero()
                };
                output_data.push(avg);
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        &[batch_size, channels, output_l],
        input.backend().clone(),
    )?)
}

/// Applies 3D max pooling over an input signal.
pub fn max_pool3d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 5D [N, C, D, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_d, input_h, input_w] = [
        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
    ];

    let stride = stride.unwrap_or(kernel_size);
    let output_d = (input_d + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let output_h = (input_h + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let output_w = (input_w + 2 * padding.2 - kernel_size.2) / stride.2 + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_d {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        let mut max_val = T::neg_infinity();
                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let d_in = od * stride.0 + kd;
                                    let h_in = oh * stride.1 + kh;
                                    let w_in = ow * stride.2 + kw;

                                    if d_in >= padding.0 && d_in < input_d + padding.0 &&
                                       h_in >= padding.1 && h_in < input_h + padding.1 &&
                                       w_in >= padding.2 && w_in < input_w + padding.2
                                    {
                                        let d_act = d_in - padding.0;
                                        let h_act = h_in - padding.1;
                                        let w_act = w_in - padding.2;
                                        let idx = ((((n * channels + c) * input_d + d_act) * input_h + h_act) * input_w) + w_act;
                                        let val = input_data[idx];
                                        if val > max_val {
                                            max_val = val;
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

    Ok(Tensor::from_vec_with_backend(
        output_data,
        &[batch_size, channels, output_d, output_h, output_w],
        input.backend().clone(),
    )?)
}

/// Applies 3D average pooling over an input signal.
pub fn avg_pool3d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 5D [N, C, D, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_d, input_h, input_w] = [
        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
    ];

    let stride = stride.unwrap_or(kernel_size);
    let output_d = (input_d + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let output_h = (input_h + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let output_w = (input_w + 2 * padding.2 - kernel_size.2) / stride.2 + 1;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_d {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        let mut sum = T::zero();
                        let mut count = 0;
                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let d_in = od * stride.0 + kd;
                                    let h_in = oh * stride.1 + kh;
                                    let w_in = ow * stride.2 + kw;

                                    if d_in >= padding.0 && d_in < input_d + padding.0 &&
                                       h_in >= padding.1 && h_in < input_h + padding.1 &&
                                       w_in >= padding.2 && w_in < input_w + padding.2
                                    {
                                        let d_act = d_in - padding.0;
                                        let h_act = h_in - padding.1;
                                        let w_act = w_in - padding.2;
                                        let idx = ((((n * channels + c) * input_d + d_act) * input_h + h_act) * input_w) + w_act;
                                        sum = sum + input_data[idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                        let avg = if count > 0 {
                            sum / T::from(count).unwrap()
                        } else {
                            T::zero()
                        };
                        output_data.push(avg);
                    }
                }
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        &[batch_size, channels, output_d, output_h, output_w],
        input.backend().clone(),
    )?)
}

/// Applies 2D adaptive average pooling over an input signal.
pub fn adaptive_avg_pool2d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: (usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 4 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_h, input_w] = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]];
    let [out_h, out_w] = [output_size.0, output_size.1];

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_h * out_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = (oh * input_h) / out_h;
                let h_end = ((oh + 1) * input_h + out_h - 1) / out_h;
                for ow in 0..out_w {
                    let w_start = (ow * input_w) / out_w;
                    let w_end = ((ow + 1) * input_w + out_w - 1) / out_w;

                    let mut sum = T::zero();
                    let count = (h_end - h_start) * (w_end - w_start);
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let idx = ((n * channels + c) * input_h + ih) * input_w + iw;
                            sum = sum + input_data[idx];
                        }
                    }
                    output_data.push(sum / T::from(count).unwrap());
                }
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_h, out_w], input.backend().clone())?)
}

/// Applies 2D adaptive max pooling over an input signal.
pub fn adaptive_max_pool2d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: (usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 4 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 4D [N, C, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_h, input_w] = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]];
    let [out_h, out_w] = [output_size.0, output_size.1];

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_h * out_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = (oh * input_h) / out_h;
                let h_end = ((oh + 1) * input_h + out_h - 1) / out_h;
                for ow in 0..out_w {
                    let w_start = (ow * input_w) / out_w;
                    let w_end = ((ow + 1) * input_w + out_w - 1) / out_w;

                    let mut max_val = T::neg_infinity();
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let idx = ((n * channels + c) * input_h + ih) * input_w + iw;
                            let val = input_data[idx];
                            if val > max_val { max_val = val; }
                        }
                    }
                    output_data.push(max_val);
                }
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_h, out_w], input.backend().clone())?)
}

/// Applies 1D adaptive average pooling over an input signal.
pub fn adaptive_avg_pool1d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: usize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 3D [N, C, L], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_l] = [input_shape[0], input_shape[1], input_shape[2]];
    let out_l = output_size;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_l);

    for n in 0..batch_size {
        for c in 0..channels {
            for ol in 0..out_l {
                let l_start = (ol * input_l) / out_l;
                let l_end = ((ol + 1) * input_l + out_l - 1) / out_l;

                let mut sum = T::zero();
                let count = l_end - l_start;
                for il in l_start..l_end {
                    let idx = (n * channels + c) * input_l + il;
                    sum = sum + input_data[idx];
                }
                output_data.push(sum / T::from(count).unwrap());
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_l], input.backend().clone())?)
}

/// Applies 1D adaptive max pooling over an input signal.
pub fn adaptive_max_pool1d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: usize,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 3 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 3D [N, C, L], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_l] = [input_shape[0], input_shape[1], input_shape[2]];
    let out_l = output_size;

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_l);

    for n in 0..batch_size {
        for c in 0..channels {
            for ol in 0..out_l {
                let l_start = (ol * input_l) / out_l;
                let l_end = ((ol + 1) * input_l + out_l - 1) / out_l;

                let mut max_val = T::neg_infinity();
                for il in l_start..l_end {
                    let idx = (n * channels + c) * input_l + il;
                    let val = input_data[idx];
                    if val > max_val { max_val = val; }
                }
                output_data.push(max_val);
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_l], input.backend().clone())?)
}

/// Applies 3D adaptive average pooling over an input signal.
pub fn adaptive_avg_pool3d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: (usize, usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 5D [N, C, D, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_d, input_h, input_w] = [input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]];
    let [out_d, out_h, out_w] = [output_size.0, output_size.1, output_size.2];

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_d * out_h * out_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_d {
                let d_start = (od * input_d) / out_d;
                let d_end = ((od + 1) * input_d + out_d - 1) / out_d;
                for oh in 0..out_h {
                    let h_start = (oh * input_h) / out_h;
                    let h_end = ((oh + 1) * input_h + out_h - 1) / out_h;
                    for ow in 0..out_w {
                        let w_start = (ow * input_w) / out_w;
                        let w_end = ((ow + 1) * input_w + out_w - 1) / out_w;

                        let mut sum = T::zero();
                        let count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let idx = (((n * channels + c) * input_d + id) * input_h + ih) * input_w + iw;
                                    sum = sum + input_data[idx];
                                }
                            }
                        }
                        output_data.push(sum / T::from(count).unwrap());
                    }
                }
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_d, out_h, out_w], input.backend().clone())?)
}

/// Applies 3D adaptive max pooling over an input signal.
pub fn adaptive_max_pool3d<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>, T>,
    output_size: (usize, usize, usize),
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() != 5 {
        return Err(NNError::InvalidInput {
            message: format!("Input must be 5D [N, C, D, H, W], got {}D", input_shape.len()),
        });
    }

    let [batch_size, channels, input_d, input_h, input_w] = [input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]];
    let [out_d, out_h, out_w] = [output_size.0, output_size.1, output_size.2];

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(batch_size * channels * out_d * out_h * out_w);

    for n in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_d {
                let d_start = (od * input_d) / out_d;
                let d_end = ((od + 1) * input_d + out_d - 1) / out_d;
                for oh in 0..out_h {
                    let h_start = (oh * input_h) / out_h;
                    let h_end = ((oh + 1) * input_h + out_h - 1) / out_h;
                    for ow in 0..out_w {
                        let w_start = (ow * input_w) / out_w;
                        let w_end = ((ow + 1) * input_w + out_w - 1) / out_w;

                        let mut max_val = T::neg_infinity();
                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let idx = (((n * channels + c) * input_d + id) * input_h + ih) * input_w + iw;
                                    let val = input_data[idx];
                                    if val > max_val { max_val = val; }
                                }
                            }
                        }
                        output_data.push(max_val);
                    }
                }
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(output_data, &[batch_size, channels, out_d, out_h, out_w], input.backend().clone())?)
}
