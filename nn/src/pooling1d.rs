//! 1D Pooling layers for neural networks.
//!
//! This module implements 1D pooling operations: MaxPool1d, AvgPool1d, and AdaptiveAvgPool1d.

use std::fmt;

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;

/// 1D Max Pooling layer.
///
/// Applies 1D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
///
/// # Shape
/// - Input: `(N, C, L_in)` where N is batch size, C is channels, L_in is input length
/// - Output: `(N, C, L_out)` where L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
///
/// # Examples
/// ```rust
/// use nn::{MaxPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = MaxPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 50]);
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride. If None, defaults to kernel_size
    pub stride: Option<usize>,
    /// Padding
    pub padding: usize,
}

impl MaxPool1d {
    /// Create a new MaxPool1d layer.
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        assert!(kernel_size > 0, "kernel_size must be > 0");
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for MaxPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 3D input (batch, channels, length), got {}D",
                    input_shape.len()
                ),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_length = input_shape[2];

        let stride = self.stride.unwrap_or(self.kernel_size);
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / stride + 1;
        let output_shape = vec![batch_size, channels, output_length];

        let mut output_data = Vec::with_capacity(batch_size * channels * output_length);
        let input_data = input.as_slice();

        for b in 0..batch_size {
            for c in 0..channels {
                for ol in 0..output_length {
                    let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                    for k in 0..self.kernel_size {
                        let input_pos = (ol * stride + k) as isize - self.padding as isize;

                        if input_pos >= 0 && input_pos < input_length as isize {
                            let idx = b * (channels * input_length)
                                + c * input_length
                                + input_pos as usize;
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

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<crate::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "MaxPool1d"
    }
}

/// 1D Average Pooling layer.
///
/// Applies 1D average pooling over an input signal.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, L_out)` where L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AvgPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 50]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride. If None, defaults to kernel_size
    pub stride: Option<usize>,
    /// Padding
    pub padding: usize,
}

impl AvgPool1d {
    /// Create a new AvgPool1d layer.
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        assert!(kernel_size > 0, "kernel_size must be > 0");
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for AvgPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 3D input (batch, channels, length), got {}D",
                    input_shape.len()
                ),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_length = input_shape[2];

        let stride = self.stride.unwrap_or(self.kernel_size);
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / stride + 1;
        let output_shape = vec![batch_size, channels, output_length];

        let mut output_data = Vec::with_capacity(batch_size * channels * output_length);
        let input_data = input.as_slice();

        for b in 0..batch_size {
            for c in 0..channels {
                for ol in 0..output_length {
                    let mut sum = T::from(0.0).unwrap();
                    let mut count = 0;

                    for k in 0..self.kernel_size {
                        let input_pos = (ol * stride + k) as isize - self.padding as isize;

                        if input_pos >= 0 && input_pos < input_length as isize {
                            let idx = b * (channels * input_length)
                                + c * input_length
                                + input_pos as usize;
                            sum = sum + input_data[idx];
                            count += 1;
                        }
                    }

                    let avg = if count > 0 {
                        sum / T::from(count as f64).unwrap()
                    } else {
                        T::from(0.0).unwrap()
                    };
                    output_data.push(avg);
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<crate::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AvgPool1d"
    }
}

/// 1D Adaptive Average Pooling layer.
///
/// Applies 1D adaptive average pooling to an input signal, outputting a fixed size.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, output_size)` where output_size is specified
///
/// # Examples
/// ```rust
/// use nn::{AdaptiveAvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool1d::new(10);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 10]);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool1d {
    /// Output size
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create a new AdaptiveAvgPool1d layer.
    pub fn new(output_size: usize) -> Self {
        assert!(output_size > 0, "output_size must be > 0");
        Self { output_size }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for AdaptiveAvgPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 3D input (batch, channels, length), got {}D",
                    input_shape.len()
                ),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_length = input_shape[2];

        let output_shape = vec![batch_size, channels, self.output_size];
        let mut output_data = Vec::with_capacity(batch_size * channels * self.output_size);
        let input_data = input.as_slice();

        for b in 0..batch_size {
            for c in 0..channels {
                for ol in 0..self.output_size {
                    // Calculate the range of input elements that contribute to this output element
                    let input_start = (ol * input_length) / self.output_size;
                    let input_end = ((ol + 1) * input_length) / self.output_size;
                    let count = input_end - input_start;

                    let mut sum = T::from(0.0).unwrap();
                    for k in input_start..input_end {
                        let idx = b * (channels * input_length) + c * input_length + k;
                        sum = sum + input_data[idx];
                    }

                    let avg = sum / T::from(count as f64).unwrap();
                    output_data.push(avg);
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<crate::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AdaptiveAvgPool1d"
    }
}

impl fmt::Display for MaxPool1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MaxPool1d(kernel_size={}, stride={:?}, padding={})",
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

impl fmt::Display for AvgPool1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AvgPool1d(kernel_size={}, stride={:?}, padding={})",
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

impl fmt::Display for AdaptiveAvgPool1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdaptiveAvgPool1d(output_size={})", self.output_size)
    }
}

