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
