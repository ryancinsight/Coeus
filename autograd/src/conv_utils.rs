//! Convolution utility functions for efficient gradient computation
//!
//! This module provides im2col and col2im operations that transform convolution
//! into matrix multiplication, enabling efficient gradient computation.

use crate::error::{AutogradError, Result};
use coeus_backend::CpuBackend;
use coeus_dtype::traits::FloatExt;
use coeus_dtype::DataType;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

/// Transform 4D input tensor into 2D column matrix for efficient convolution.
///
/// Im2col (image to column) is a standard technique that transforms the convolution
/// operation into a matrix multiplication. For each output position, it extracts
/// the corresponding input patch and arranges it as a column in the output matrix.
///
/// # Arguments
/// * `input` - Input tensor of shape `[batch, in_channels, height, width]`
/// * `kernel_h` - Kernel height
/// * `kernel_w` - Kernel width
/// * `stride_h` - Stride in height dimension
/// * `stride_w` - Stride in width dimension
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// Column matrix of shape `[in_channels * kernel_h * kernel_w, batch * out_h * out_w]`
///
/// # Panics
/// Panics if input is not 4D or if dimensions are invalid.
pub fn im2col<T: DataType + FloatExt>(
    input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(AutogradError::InvalidOperation {
            operation: format!("im2col requires 4D input, got {}D", input_shape.len()),
        });
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    // Compute output dimensions
    let out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;

    // Column matrix dimensions
    let col_h = in_channels * kernel_h * kernel_w;
    let col_w = batch * out_h * out_w;

    let mut col_data = vec![T::zero(); col_h * col_w];
    let input_data = input.as_slice();

    // For each output position
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                // Column index in output matrix
                let col_idx = (b * out_h * out_w) + (oh * out_w) + ow;

                // Extract patch from input
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            // Input position (with padding)
                            let ih = (oh * stride_h + kh) as isize - padding_h as isize;
                            let iw = (ow * stride_w + kw) as isize - padding_w as isize;

                            // Row index in output matrix
                            let row_idx = (c * kernel_h * kernel_w) + (kh * kernel_w) + kw;

                            // Check bounds (padding)
                            let value =
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let input_idx = ((b * in_channels + c) * in_h + ih as usize)
                                        * in_w
                                        + iw as usize;
                                    input_data[input_idx]
                                } else {
                                    T::zero()
                                };

                            col_data[row_idx * col_w + col_idx] = value;
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(col_data, &[col_h, col_w]).map_err(Into::into)
}

/// Transform 2D column matrix back into 4D input tensor (inverse of im2col).
///
/// Col2im (column to image) is the inverse operation of im2col. It accumulates
/// the columns back into the input tensor format, handling overlapping patches
/// by summing their contributions.
///
/// # Arguments
/// * `col` - Column matrix of shape `[in_channels * kernel_h * kernel_w, batch * out_h * out_w]`
/// * `input_shape` - Target input shape `[batch, in_channels, height, width]`
/// * `kernel_h` - Kernel height
/// * `kernel_w` - Kernel width
/// * `stride_h` - Stride in height dimension
/// * `stride_w` - Stride in width dimension
/// * `padding_h` - Padding in height dimension
/// * `padding_w` - Padding in width dimension
///
/// # Returns
/// Input tensor of shape `[batch, in_channels, height, width]`
///
/// # Panics
/// Panics if column matrix dimensions are invalid.
pub fn col2im<T: DataType + FloatExt>(
    col: &Tensor<CpuBackend, DenseStorage<T>, T>,
    input_shape: &[usize],
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
    if input_shape.len() != 4 {
        return Err(AutogradError::InvalidOperation {
            operation: format!("col2im requires 4D input shape, got {}D", input_shape.len()),
        });
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    // Compute output dimensions
    let out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;

    let col_shape = col.shape().dims();
    let col_h = col_shape[0];
    let col_w = col_shape[1];

    // Validate dimensions
    if col_h != in_channels * kernel_h * kernel_w {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "col2im: column height mismatch: expected {}, got {}",
                in_channels * kernel_h * kernel_w,
                col_h
            ),
        });
    }

    if col_w != batch * out_h * out_w {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "col2im: column width mismatch: expected {}, got {}",
                batch * out_h * out_w,
                col_w
            ),
        });
    }

    let mut input_data = vec![T::zero(); batch * in_channels * in_h * in_w];
    let col_data = col.as_slice();

    // Accumulate columns back into input
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                // Column index in input matrix
                let col_idx = (b * out_h * out_w) + (oh * out_w) + ow;

                // Accumulate patch into input
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            // Input position (with padding)
                            let ih = (oh * stride_h + kh) as isize - padding_h as isize;
                            let iw = (ow * stride_w + kw) as isize - padding_w as isize;

                            // Row index in input matrix
                            let row_idx = (c * kernel_h * kernel_w) + (kh * kernel_w) + kw;

                            // Check bounds (padding)
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let input_idx = ((b * in_channels + c) * in_h + ih as usize) * in_w
                                    + iw as usize;
                                input_data[input_idx] =
                                    input_data[input_idx] + col_data[row_idx * col_w + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(input_data, input_shape).map_err(Into::into)
}

/// Compute 4D tensor reduction by summing over specified axes.
///
/// This function reduces a 4D tensor by summing over the specified axes.
/// For `Conv2D` bias gradients, we sum over axes [0, 2, 3] (batch, height, width).
///
/// # Arguments
/// * `tensor` - Input tensor of shape `[batch, channels, height, width]`
/// * `axes` - Axes to sum over (e.g., `[0, 2, 3]` for bias gradient)
///
/// # Returns
/// Reduced tensor with summed dimensions removed
///
/// # Panics
/// Panics if tensor is not 4D or if axes are invalid.
pub fn reduce_sum_4d<T: DataType + FloatExt>(
    tensor: &Tensor<CpuBackend, DenseStorage<T>, T>,
    axes: &[usize],
) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
    let shape = tensor.shape().dims();
    if shape.len() != 4 {
        return Err(AutogradError::InvalidOperation {
            operation: format!("reduce_sum_4d requires 4D tensor, got {}D", shape.len()),
        });
    }

    // For Conv2D bias gradient: sum over [0, 2, 3] -> result shape [channels]
    if axes == [0, 2, 3] {
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let mut result = vec![T::zero(); channels];
        let data = tensor.as_slice();

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        result[c] = result[c] + data[idx];
                    }
                }
            }
        }

        return Tensor::from_vec(result, &[channels]).map_err(Into::into);
    }

    Err(AutogradError::InvalidOperation {
        operation: format!("reduce_sum_4d: unsupported axes {axes:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_im2col_basic() {
        // Test basic im2col operation
        // Input: [1, 1, 3, 3]
        let input = Tensor::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
                Float32::new(9.0),
            ],
            &[1, 1, 3, 3],
        )
        .unwrap();

        // Kernel 2x2, stride 1, no padding
        let col = im2col(&input, 2, 2, 1, 1, 0, 0).unwrap();

        // Expected shape: [1*2*2, 1*2*2] = [4, 4]
        assert_eq!(col.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_col2im_basic() {
        // Test col2im operation (inverse of im2col)
        let input = Tensor::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
                Float32::new(9.0),
            ],
            &[1, 1, 3, 3],
        )
        .unwrap();

        // im2col -> col2im should preserve shape
        let col = im2col(&input, 2, 2, 1, 1, 0, 0).unwrap();
        let reconstructed = col2im(&col, &[1, 1, 3, 3], 2, 2, 1, 1, 0, 0).unwrap();

        assert_eq!(reconstructed.shape().dims(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_reduce_sum_4d() {
        // Test 4D reduction for bias gradient
        // Input: [2, 3, 2, 2] (batch=2, channels=3, height=2, width=2)
        let input = Tensor::from_vec(vec![Float32::new(1.0); 24], &[2, 3, 2, 2]).unwrap();

        // Sum over [0, 2, 3] -> result shape [3]
        let result = reduce_sum_4d(&input, &[0, 2, 3]).unwrap();

        assert_eq!(result.shape().dims(), &[3]);
        // Each channel should have sum = 2*2*2 = 8
        for i in 0..3 {
            assert_eq!(result.as_slice()[i].get(), 8.0);
        }
    }
}
