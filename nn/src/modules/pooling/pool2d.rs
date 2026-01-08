//! 2D Pooling layers for neural networks.
//!
//! This module implements 2D pooling operations: MaxPool2d, AvgPool2d,
//! AdaptiveAvgPool2d, and AdaptiveMaxPool2d.

use std::fmt;

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;

/// 2D Max Pooling layer.
///
/// Applies 2D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
///
/// # Shape
/// - Input: `(N, C, H_in, W_in)` where N is batch size, C is channels, H_in/W_in are spatial dims
/// - Output: `(N, C, H_out, W_out)` where H_out/W_out depend on kernel_size, stride, padding
///
/// # Examples
/// ```rust
/// use nn::{MaxPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (height, width)
    /// * `stride` - Stride (height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (height, width)
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let h_out = (input_h + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let w_out = (input_w + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        (h_out, w_out)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for MaxPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Input must be 4D [N, C, H_in, W_in], got {}D", input_shape.len()),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let (output_h, output_w) = self.output_size(input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                        // Find max in pooling window
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h_in = out_h * stride.0 + kh;
                                let w_in = out_w * stride.1 + kw;

                                // Handle padding (treat as -inf for max pooling)
                                if h_in >= self.padding.0
                                    && h_in < input_h + self.padding.0
                                    && w_in >= self.padding.1
                                    && w_in < input_w + self.padding.1
                                {
                                    let h_actual = h_in - self.padding.0;
                                    let w_actual = w_in - self.padding.1;

                                    if h_actual < input_h && w_actual < input_w {
                                        let input_idx = ((n * channels + c) * input_h + h_actual)
                                            * input_w
                                            + w_actual;
                                        let val = input_data[input_idx];
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
        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "MaxPool2d"
    }
}

/// Placeholder for AvgPool2d - to be implemented
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for AvgPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Input must be 4D [N, C, H_in, W_in], got {}D", input_shape.len()),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let (output_h, output_w) = self.output_size(input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        let mut sum = T::zero();
                        let mut count = 0;

                        // Sum values in pooling window
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h_in = out_h * stride.0 + kh;
                                let w_in = out_w * stride.1 + kw;

                                // Handle padding (treat as zero for average pooling)
                                if h_in >= self.padding.0
                                    && h_in < input_h + self.padding.0
                                    && w_in >= self.padding.1
                                    && w_in < input_w + self.padding.1
                                {
                                    let h_actual = h_in - self.padding.0;
                                    let w_actual = w_in - self.padding.1;

                                    if h_actual < input_h && w_actual < input_w {
                                        let input_idx = ((n * channels + c) * input_h + h_actual)
                                            * input_w
                                            + w_actual;
                                        sum = sum + input_data[input_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        // Compute average
                        let avg_val = if count > 0 {
                            sum / T::from(count).unwrap()
                        } else {
                            T::zero()
                        };

                        output_data.push(avg_val);
                    }
                }
            }
        }

        let output_shape = &[batch_size, channels, output_h, output_w];
        Tensor::from_vec(output_data, output_shape)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AvgPool2d"
    }
}

/// Placeholder for AdaptiveAvgPool2d - to be implemented
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    /// Output size (height, width)
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for AdaptiveAvgPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Input must be 4D [N, C, H_in, W_in], got {}D", input_shape.len()),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let output_h = self.output_size.0;
        let output_w = self.output_size.1;

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        // Calculate the input region for this output pixel
                        let h_start = (out_h * input_h) / output_h;
                        let h_end = ((out_h + 1) * input_h) / output_h;
                        let w_start = (out_w * input_w) / output_w;
                        let w_end = ((out_w + 1) * input_w) / output_w;

                        let mut sum = T::zero();
                        let mut count = 0;

                        // Sum values in the adaptive region
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                                sum = sum + input_data[input_idx];
                                count += 1;
                            }
                        }

                        // Compute average
                        let avg_val = if count > 0 {
                            sum / T::from(count).unwrap()
                        } else {
                            T::zero()
                        };

                        output_data.push(avg_val);
                    }
                }
            }
        }

        let output_shape = &[batch_size, channels, output_h, output_w];
        Tensor::from_vec(output_data, output_shape)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AdaptiveAvgPool2d"
    }
}

/// Placeholder for AdaptiveMaxPool2d - to be implemented
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    /// Output size (height, width)
    pub output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for AdaptiveMaxPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Input must be 4D [N, C, H_in, W_in], got {}D", input_shape.len()),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let output_h = self.output_size.0;
        let output_w = self.output_size.1;

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        // Calculate the input region for this output pixel
                        let h_start = (out_h * input_h) / output_h;
                        let h_end = ((out_h + 1) * input_h) / output_h;
                        let w_start = (out_w * input_w) / output_w;
                        let w_end = ((out_w + 1) * input_w) / output_w;

                        let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                        // Find max in the adaptive region
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                                let val = input_data[input_idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }

                        output_data.push(max_val);
                    }
                }
            }
        }

        let output_shape = &[batch_size, channels, output_h, output_w];
        Tensor::from_vec(output_data, output_shape)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AdaptiveMaxPool2d"
    }
}

impl fmt::Display for MaxPool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MaxPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

impl fmt::Display for AvgPool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AvgPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

impl fmt::Display for AdaptiveAvgPool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdaptiveAvgPool2d(output_size={:?})",
            self.output_size
        )
    }
}

impl fmt::Display for AdaptiveMaxPool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdaptiveMaxPool2d(output_size={:?})",
            self.output_size
        )
    }
}

