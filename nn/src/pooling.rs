//! Pooling layers for neural networks (1D and 2D).

use coeus_backend::CpuBackend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

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
/// use coeus_nn::{MaxPool1d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pool = MaxPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
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

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for MaxPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3 {
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

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
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
/// use coeus_nn::{AvgPool1d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pool = AvgPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
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

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for AvgPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3 {
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
                    let mut sum = T::zero();
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
                        T::zero()
                    };

                    output_data.push(avg);
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
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

/// Adaptive Average Pooling 1D layer.
///
/// Applies adaptive 1D average pooling. The output size is specified,
/// and the layer automatically computes the kernel size and stride.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, L_out)` where L_out is specified by output_size
///
/// # Examples
/// ```rust
/// use coeus_nn::{AdaptiveAvgPool1d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool1d::new(10); // Output length = 10
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
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
        Self { output_size }
    }

    /// Compute adaptive pooling parameters.
    fn compute_adaptive_params(
        input_size: usize,
        output_size: usize,
        output_idx: usize,
    ) -> (usize, usize) {
        let start = (output_idx * input_size) / output_size;
        let end = ((output_idx + 1) * input_size) / output_size;
        (start, end)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for AdaptiveAvgPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3 {
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
                    let (start, end) =
                        Self::compute_adaptive_params(input_length, self.output_size, ol);

                    let mut sum = T::zero();
                    let mut count = 0;

                    for pos in start..end {
                        let idx = b * (channels * input_length) + c * input_length + pos;
                        sum = sum + input_data[idx];
                        count += 1;
                    }

                    let avg = if count > 0 {
                        sum / T::from(count as f64).unwrap()
                    } else {
                        T::zero()
                    };

                    output_data.push(avg);
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
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

/// 2D Max Pooling layer.
///
/// Applies a 2D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
///
/// Input shape: [N, C, H_in, W_in]
/// Output shape: [N, C, H_out, W_out]
///
/// where:
/// - H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
/// - W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{MaxPool2d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create MaxPool2d with 2x2 kernel, stride 2
/// let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: [2, 64, 16, 16] (downsampled by 2x)
/// let output = <MaxPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 16]);
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

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for MaxPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 4, "Input must be 4D [N, C, H_in, W_in]");

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

        Tensor::from_vec(output_data, &[batch_size, channels, output_h, output_w])
            .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        vec![] // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior
    }

    fn name(&self) -> &str {
        "MaxPool2d"
    }
}

/// 2D Average Pooling layer.
///
/// Applies a 2D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
///
/// Input shape: [N, C, H_in, W_in]
/// Output shape: [N, C, H_out, W_out]
///
/// where:
/// - H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
/// - W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{AvgPool2d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create AvgPool2d with 2x2 kernel, stride 2
/// let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: [2, 64, 16, 16] (downsampled by 2x)
/// let output = <AvgPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 16]);
/// ```
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
    /// Create a new AvgPool2d layer.
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

impl<T: DataType + FloatExt> Module<CpuBackend, DenseStorage<T>, T> for AvgPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 4, "Input must be 4D [N, C, H_in, W_in]");

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let (output_h, output_w) = self.output_size(input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        let kernel_area = T::from((self.kernel_size.0 * self.kernel_size.1) as f64).unwrap();

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        let mut sum = T::zero();

                        // Compute average in pooling window
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h_in = out_h * stride.0 + kh;
                                let w_in = out_w * stride.1 + kw;

                                // Handle padding (treat as 0 for average pooling)
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
                                    }
                                }
                            }
                        }

                        output_data.push(sum / kernel_area);
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, channels, output_h, output_w])
            .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        vec![] // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior
    }

    fn name(&self) -> &str {
        "AvgPool2d"
    }
}

/// Adaptive Average Pooling 2D layer.
///
/// Applies adaptive 2D average pooling over an input signal composed of several input planes.
/// The output size is specified, and the layer automatically computes the kernel size and stride
/// to produce the desired output dimensions.
///
/// This is essential for modern CNN architectures like ResNet and EfficientNet that need
/// resolution-agnostic feature extraction.
///
/// # Shape
/// - Input: `(N, C, H_in, W_in)` or `(C, H_in, W_in)`
/// - Output: `(N, C, H_out, W_out)` or `(C, H_out, W_out)` where `H_out` and `W_out` are specified by `output_size`
///
/// # Examples
/// ```rust
/// use coeus_nn::{AdaptiveAvgPool2d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 3, 224, 224]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
/// ```
///
/// # References
/// - He et al. (2015): "Deep Residual Learning for Image Recognition" - Uses adaptive pooling in ResNet
/// - Tan & Le (2019): "EfficientNet: Rethinking Model Scaling for CNNs" - Uses adaptive pooling
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    /// Output size (height, width)
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d layer.
    ///
    /// # Arguments
    /// * `output_size` - The target output size (height, width)
    ///
    /// # Examples
    /// ```rust
    /// use coeus_nn::AdaptiveAvgPool2d;
    ///
    /// let pool = AdaptiveAvgPool2d::new((7, 7)); // Output will be 7x7
    /// let global_pool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
    /// ```
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Compute adaptive pooling parameters for a given dimension.
    ///
    /// Returns (start_index, end_index) for the pooling window.
    fn compute_adaptive_params(
        input_size: usize,
        output_size: usize,
        output_idx: usize,
    ) -> (usize, usize) {
        let start = (output_idx * input_size) / output_size;
        let end = ((output_idx + 1) * input_size) / output_size;
        (start, end)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for AdaptiveAvgPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        // Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
        let (batch_size, channels, height, width) = match input_shape.len() {
            3 => (1, input_shape[0], input_shape[1], input_shape[2]),
            4 => (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            _ => {
                return Err(NNError::InvalidInput {
                    message: format!("Expected 3D or 4D input, got {}D", input_shape.len()),
                })
            }
        };

        let (out_height, out_width) = self.output_size;
        let output_shape = if input_shape.len() == 3 {
            vec![channels, out_height, out_width]
        } else {
            vec![batch_size, channels, out_height, out_width]
        };

        let mut output_data = Vec::with_capacity(batch_size * channels * out_height * out_width);

        // Reshape input to 4D for uniform processing
        let input_data = input.as_slice();

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        // Compute adaptive pooling window
                        let (h_start, h_end) =
                            Self::compute_adaptive_params(height, out_height, oh);
                        let (w_start, w_end) = Self::compute_adaptive_params(width, out_width, ow);

                        // Compute average over the window
                        let mut sum = T::zero();
                        let mut count = 0;

                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w;
                                sum = sum + input_data[idx];
                                count += 1;
                            }
                        }

                        let avg = if count > 0 {
                            sum / T::from(count as f64).unwrap()
                        } else {
                            T::zero()
                        };

                        output_data.push(avg);
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        Vec::new() // No learnable parameters
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

/// Adaptive Max Pooling 2D layer.
///
/// Applies adaptive 2D max pooling over an input signal composed of several input planes.
/// The output size is specified, and the layer automatically computes the kernel size and stride
/// to produce the desired output dimensions.
///
/// # Shape
/// - Input: `(N, C, H_in, W_in)` or `(C, H_in, W_in)`
/// - Output: `(N, C, H_out, W_out)` or `(C, H_out, W_out)` where `H_out` and `W_out` are specified by `output_size`
///
/// # Examples
/// ```rust
/// use coeus_nn::{AdaptiveMaxPool2d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pool = AdaptiveMaxPool2d::new((1, 1)); // Global max pooling
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 3, 224, 224]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    /// Output size (height, width)
    pub output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    /// Create a new AdaptiveMaxPool2d layer.
    ///
    /// # Arguments
    /// * `output_size` - The target output size (height, width)
    ///
    /// # Examples
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool2d;
    ///
    /// let pool = AdaptiveMaxPool2d::new((7, 7)); // Output will be 7x7
    /// let global_pool = AdaptiveMaxPool2d::new((1, 1)); // Global max pooling
    /// ```
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Compute adaptive pooling parameters for a given dimension.
    fn compute_adaptive_params(
        input_size: usize,
        output_size: usize,
        output_idx: usize,
    ) -> (usize, usize) {
        let start = (output_idx * input_size) / output_size;
        let end = ((output_idx + 1) * input_size) / output_size;
        (start, end)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for AdaptiveMaxPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        // Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
        let (batch_size, channels, height, width) = match input_shape.len() {
            3 => (1, input_shape[0], input_shape[1], input_shape[2]),
            4 => (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            _ => {
                return Err(NNError::InvalidInput {
                    message: format!("Expected 3D or 4D input, got {}D", input_shape.len()),
                })
            }
        };

        let (out_height, out_width) = self.output_size;
        let output_shape = if input_shape.len() == 3 {
            vec![channels, out_height, out_width]
        } else {
            vec![batch_size, channels, out_height, out_width]
        };

        let mut output_data = Vec::with_capacity(batch_size * channels * out_height * out_width);

        let input_data = input.as_slice();

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        // Compute adaptive pooling window
                        let (h_start, h_end) =
                            Self::compute_adaptive_params(height, out_height, oh);
                        let (w_start, w_end) = Self::compute_adaptive_params(width, out_width, ow);

                        // Compute max over the window
                        let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w;
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
        }

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        Vec::new() // No learnable parameters
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

/// 3D Max Pooling layer.
///
/// Applies a 3D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
/// Essential for video processing and 3D medical imaging.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)` where N is batch size, C is channels
/// - Output: `(N, C, D_out, H_out, W_out)` where:
///   - D_out = floor((D_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
///   - H_out = floor((H_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///   - W_out = floor((W_in + 2*padding[2] - kernel_size[2]) / stride[2] + 1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{MaxPool3d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create MaxPool3d with 2x2x2 kernel, stride 2
/// let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
///
/// // Input: [batch_size=1, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: [1, 64, 8, 16, 16] (downsampled by 2x in all dimensions)
/// let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 8, 16, 16]);
/// ```
///
/// # References
/// - Tran et al. (2015): "Learning Spatiotemporal Features with 3D Convolutional Networks" (C3D)
/// - Carreira & Zisserman (2017): "Quo Vadis, Action Recognition?" (I3D)
/// - Çiçek et al. (2016): "3D U-Net: Learning Dense Volumetric Segmentation"
#[derive(Debug, Clone)]
pub struct MaxPool3d {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize, usize)>,
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
}

impl MaxPool3d {
    /// Create a new MaxPool3d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (depth, height, width)
    /// * `stride` - Stride (depth, height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (depth, height, width)
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0 && s.2 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, input_d: usize, input_h: usize, input_w: usize) -> (usize, usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let d_out = (input_d + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let h_out = (input_h + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        let w_out = (input_w + 2 * self.padding.2 - self.kernel_size.2) / stride.2 + 1;
        (d_out, h_out, w_out)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for MaxPool3d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        // Input: [N, C, D_in, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(
            input_shape.len(),
            5,
            "Input must be 5D [N, C, D_in, H_in, W_in]"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        let (output_d, output_h, output_w) = self.output_size(input_d, input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data =
            Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_d in 0..output_d {
                    for out_h in 0..output_h {
                        for out_w in 0..output_w {
                            let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                            // Find max in pooling window
                            for kd in 0..self.kernel_size.0 {
                                for kh in 0..self.kernel_size.1 {
                                    for kw in 0..self.kernel_size.2 {
                                        let d_in = out_d * stride.0 + kd;
                                        let h_in = out_h * stride.1 + kh;
                                        let w_in = out_w * stride.2 + kw;

                                        // Handle padding (treat as -inf for max pooling)
                                        if d_in >= self.padding.0
                                            && d_in < input_d + self.padding.0
                                            && h_in >= self.padding.1
                                            && h_in < input_h + self.padding.1
                                            && w_in >= self.padding.2
                                            && w_in < input_w + self.padding.2
                                        {
                                            let d_actual = d_in - self.padding.0;
                                            let h_actual = h_in - self.padding.1;
                                            let w_actual = w_in - self.padding.2;

                                            if d_actual < input_d
                                                && h_actual < input_h
                                                && w_actual < input_w
                                            {
                                                let input_idx = (((n * channels + c) * input_d
                                                    + d_actual)
                                                    * input_h
                                                    + h_actual)
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
                            }

                            output_data.push(max_val);
                        }
                    }
                }
            }
        }

        Tensor::from_vec(
            output_data,
            &[batch_size, channels, output_d, output_h, output_w],
        )
        .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        Vec::new() // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "MaxPool3d"
    }
}

/// 3D Average Pooling layer.
///
/// Applies a 3D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
/// Essential for video processing and 3D medical imaging.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)` where N is batch size, C is channels
/// - Output: `(N, C, D_out, H_out, W_out)` where:
///   - D_out = floor((D_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
///   - H_out = floor((H_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///   - W_out = floor((W_in + 2*padding[2] - kernel_size[2]) / stride[2] + 1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{AvgPool3d, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create AvgPool3d with 2x2x2 kernel, stride 2
/// let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
///
/// // Input: [batch_size=1, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: [1, 64, 8, 16, 16] (downsampled by 2x in all dimensions)
/// let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 8, 16, 16]);
/// ```
///
/// # References
/// - Tran et al. (2015): "Learning Spatiotemporal Features with 3D Convolutional Networks" (C3D)
/// - Carreira & Zisserman (2017): "Quo Vadis, Action Recognition?" (I3D)
/// - Çiçek et al. (2016): "3D U-Net: Learning Dense Volumetric Segmentation"
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize, usize)>,
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
}

impl AvgPool3d {
    /// Create a new AvgPool3d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (depth, height, width)
    /// * `stride` - Stride (depth, height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (depth, height, width)
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0 && s.2 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, input_d: usize, input_h: usize, input_w: usize) -> (usize, usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let d_out = (input_d + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let h_out = (input_h + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        let w_out = (input_w + 2 * self.padding.2 - self.kernel_size.2) / stride.2 + 1;
        (d_out, h_out, w_out)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend, DenseStorage<T>, T> for AvgPool3d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        // Input: [N, C, D_in, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(
            input_shape.len(),
            5,
            "Input must be 5D [N, C, D_in, H_in, W_in]"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        let (output_d, output_h, output_w) = self.output_size(input_d, input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data =
            Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_d in 0..output_d {
                    for out_h in 0..output_h {
                        for out_w in 0..output_w {
                            let mut sum = T::zero();
                            let mut count = 0;

                            // Compute average in pooling window
                            for kd in 0..self.kernel_size.0 {
                                for kh in 0..self.kernel_size.1 {
                                    for kw in 0..self.kernel_size.2 {
                                        let d_in = out_d * stride.0 + kd;
                                        let h_in = out_h * stride.1 + kh;
                                        let w_in = out_w * stride.2 + kw;

                                        // Handle padding (treat as 0 for average pooling)
                                        if d_in >= self.padding.0
                                            && d_in < input_d + self.padding.0
                                            && h_in >= self.padding.1
                                            && h_in < input_h + self.padding.1
                                            && w_in >= self.padding.2
                                            && w_in < input_w + self.padding.2
                                        {
                                            let d_actual = d_in - self.padding.0;
                                            let h_actual = h_in - self.padding.1;
                                            let w_actual = w_in - self.padding.2;

                                            if d_actual < input_d
                                                && h_actual < input_h
                                                && w_actual < input_w
                                            {
                                                let input_idx = (((n * channels + c) * input_d
                                                    + d_actual)
                                                    * input_h
                                                    + h_actual)
                                                    * input_w
                                                    + w_actual;
                                                sum = sum + input_data[input_idx];
                                                count += 1;
                                            }
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

        Tensor::from_vec(
            output_data,
            &[batch_size, channels, output_d, output_h, output_w],
        )
        .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
        Vec::new() // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AvgPool3d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;
    use num_traits::ToPrimitive;

    #[test]
    fn test_maxpool2d_constructor() {
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.stride, Some((2, 2)));
        assert_eq!(pool.padding, (0, 0));
    }

    #[test]
    fn test_maxpool2d_forward_shape() {
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));

        // Input: [batch_size=2, channels=3, height=4, width=4]
        let input_data: Vec<Float32> = (0..96).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[2, 3, 4, 4],
        )
        .unwrap();

        let output = <MaxPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [2, 3, 2, 2] (downsampled by 2x)
        assert_eq!(output.shape().dims(), &[2, 3, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_forward_correctness() {
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));

        // Input: [1, 1, 4, 4] with known values
        let input_data: Vec<Float32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .iter()
        .map(|&x| Float32::new(x))
        .collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 4, 4],
        )
        .unwrap();

        let output = <MaxPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
        let output_data = output.as_slice();

        // Expected output: max of each 2x2 window
        // Top-left: max(1,2,5,6) = 6
        // Top-right: max(3,4,7,8) = 8
        // Bottom-left: max(9,10,13,14) = 14
        // Bottom-right: max(11,12,15,16) = 16
        assert_eq!(output_data[0].to_f64().unwrap(), 6.0);
        assert_eq!(output_data[1].to_f64().unwrap(), 8.0);
        assert_eq!(output_data[2].to_f64().unwrap(), 14.0);
        assert_eq!(output_data[3].to_f64().unwrap(), 16.0);
    }

    #[test]
    fn test_maxpool2d_stride_default() {
        // When stride is None, it should default to kernel_size
        let pool = MaxPool2d::new((2, 2), None, (0, 0));

        let input_data: Vec<Float32> = (0..16).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 4, 4],
        )
        .unwrap();

        let output = <MaxPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [1, 1, 2, 2] (stride defaults to kernel_size)
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "kernel_size must be > 0")]
    fn test_maxpool2d_invalid_kernel_size() {
        let _pool = MaxPool2d::new((0, 2), Some((2, 2)), (0, 0));
    }

    #[test]
    fn test_avgpool2d_constructor() {
        let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.stride, Some((2, 2)));
        assert_eq!(pool.padding, (0, 0));
    }

    #[test]
    fn test_avgpool2d_forward_shape() {
        let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));

        // Input: [batch_size=2, channels=3, height=4, width=4]
        let input_data: Vec<Float32> = (0..96).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[2, 3, 4, 4],
        )
        .unwrap();

        let output = <AvgPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [2, 3, 2, 2] (downsampled by 2x)
        assert_eq!(output.shape().dims(), &[2, 3, 2, 2]);
    }

    #[test]
    fn test_avgpool2d_forward_correctness() {
        let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));

        // Input: [1, 1, 4, 4] with known values
        let input_data: Vec<Float32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .iter()
        .map(|&x| Float32::new(x))
        .collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 4, 4],
        )
        .unwrap();

        let output = <AvgPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
        let output_data = output.as_slice();

        // Expected output: average of each 2x2 window
        // Top-left: avg(1,2,5,6) = 3.5
        // Top-right: avg(3,4,7,8) = 5.5
        // Bottom-left: avg(9,10,13,14) = 11.5
        // Bottom-right: avg(11,12,15,16) = 13.5
        assert!((output_data[0].to_f64().unwrap() - 3.5).abs() < 1e-6);
        assert!((output_data[1].to_f64().unwrap() - 5.5).abs() < 1e-6);
        assert!((output_data[2].to_f64().unwrap() - 11.5).abs() < 1e-6);
        assert!((output_data[3].to_f64().unwrap() - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_avgpool2d_stride_default() {
        // When stride is None, it should default to kernel_size
        let pool = AvgPool2d::new((2, 2), None, (0, 0));

        let input_data: Vec<Float32> = (0..16).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 4, 4],
        )
        .unwrap();

        let output = <AvgPool2d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [1, 1, 2, 2] (stride defaults to kernel_size)
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "kernel_size must be > 0")]
    fn test_avgpool2d_invalid_kernel_size() {
        let _pool = AvgPool2d::new((0, 2), Some((2, 2)), (0, 0));
    }

    #[test]
    fn test_adaptive_avgpool2d_constructor() {
        let pool: AdaptiveAvgPool2d = AdaptiveAvgPool2d::new((7, 7));
        assert_eq!(pool.output_size, (7, 7));
    }

    #[test]
    fn test_adaptive_avgpool2d_forward_shape() {
        let pool = AdaptiveAvgPool2d::new((7, 7));
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 3, 14, 14]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3, 7, 7]);
    }

    #[test]
    fn test_adaptive_avgpool2d_global_pooling() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
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

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);

        // Global average should be (1+2+3+4+5+6+7+8+9)/9 = 5.0
        let expected = 5.0;
        assert!((output.as_slice()[0].get() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avgpool2d_3d_input() {
        let pool = AdaptiveAvgPool2d::new((2, 2));
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[3, 4, 4]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2, 2]);
    }

    #[test]
    fn test_adaptive_maxpool2d_constructor() {
        let pool: AdaptiveMaxPool2d = AdaptiveMaxPool2d::new((7, 7));
        assert_eq!(pool.output_size, (7, 7));
    }

    #[test]
    fn test_adaptive_maxpool2d_forward_shape() {
        let pool = AdaptiveMaxPool2d::new((7, 7));
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 3, 14, 14]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3, 7, 7]);
    }

    #[test]
    fn test_adaptive_maxpool2d_global_pooling() {
        let pool = AdaptiveMaxPool2d::new((1, 1));
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(9.0),
                Float32::new(7.0),
                Float32::new(8.0),
                Float32::new(6.0),
            ],
            &[1, 1, 3, 3],
        )
        .unwrap();

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);

        // Global max should be 9.0
        let expected = 9.0;
        assert!((output.as_slice()[0].get() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_maxpool2d_3d_input() {
        let pool = AdaptiveMaxPool2d::new((2, 2));
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[3, 4, 4]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2, 2]);
    }

    #[test]
    fn test_maxpool1d_creation() {
        let pool = MaxPool1d::new(2, Some(2), 0);
        assert_eq!(pool.kernel_size, 2);
    }

    #[test]
    fn test_maxpool1d_forward_basic() {
        let pool = MaxPool1d::new(2, Some(2), 0);
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 64, 50]);
    }

    #[test]
    fn test_maxpool1d_forward_with_stride() {
        let pool = MaxPool1d::new(3, Some(2), 0);
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 32, 100]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 32, 49]);
    }

    #[test]
    fn test_maxpool1d_forward_computation() {
        let pool = MaxPool1d::new(2, Some(2), 0);
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(input_data, &[1, 1, 4])
                .unwrap();
        let output = pool.forward(&input).unwrap();

        // Expected: max([1, 2]) = 2, max([3, 4]) = 4
        assert_eq!(output.shape().dims(), &[1, 1, 2]);
        assert_eq!(output.as_slice()[0].get(), 2.0);
        assert_eq!(output.as_slice()[1].get(), 4.0);
    }

    #[test]
    fn test_avgpool1d_creation() {
        let pool = AvgPool1d::new(2, Some(2), 0);
        assert_eq!(pool.kernel_size, 2);
    }

    #[test]
    fn test_avgpool1d_forward_basic() {
        let pool = AvgPool1d::new(2, Some(2), 0);
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 64, 50]);
    }

    #[test]
    fn test_avgpool1d_forward_computation() {
        let pool = AvgPool1d::new(2, Some(2), 0);
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(3.0),
            Float32::new(2.0),
            Float32::new(4.0),
        ];
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(input_data, &[1, 1, 4])
                .unwrap();
        let output = pool.forward(&input).unwrap();

        // Expected: avg([1, 3]) = 2, avg([2, 4]) = 3
        assert_eq!(output.shape().dims(), &[1, 1, 2]);
        assert_eq!(output.as_slice()[0].get(), 2.0);
        assert_eq!(output.as_slice()[1].get(), 3.0);
    }

    #[test]
    fn test_adaptive_avgpool1d_creation() {
        let pool = AdaptiveAvgPool1d::new(10);
        assert_eq!(pool.output_size, 10);
    }

    #[test]
    fn test_adaptive_avgpool1d_forward_basic() {
        let pool = AdaptiveAvgPool1d::new(10);
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 64, 10]);
    }

    #[test]
    fn test_adaptive_avgpool1d_forward_upsampling() {
        let pool = AdaptiveAvgPool1d::new(20);
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 32, 10]).unwrap();
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 32, 20]);
    }

    #[test]
    fn test_adaptive_avgpool1d_forward_computation() {
        let pool = AdaptiveAvgPool1d::new(2);
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(input_data, &[1, 1, 4])
                .unwrap();
        let output = pool.forward(&input).unwrap();

        // Expected: avg([1, 2]) = 1.5, avg([3, 4]) = 3.5
        assert_eq!(output.shape().dims(), &[1, 1, 2]);
        assert_eq!(output.as_slice()[0].get(), 1.5);
        assert_eq!(output.as_slice()[1].get(), 3.5);
    }

    #[test]
    fn test_maxpool3d_constructor() {
        let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
        assert_eq!(pool.kernel_size, (2, 2, 2));
        assert_eq!(pool.stride, Some((2, 2, 2)));
        assert_eq!(pool.padding, (0, 0, 0));
    }

    #[test]
    fn test_maxpool3d_forward_shape() {
        let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Input: [batch_size=1, channels=3, depth=4, height=4, width=4]
        let input_data: Vec<Float32> = (0..192).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 3, 4, 4, 4],
        )
        .unwrap();

        let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [1, 3, 2, 2, 2] (downsampled by 2x in all dimensions)
        assert_eq!(output.shape().dims(), &[1, 3, 2, 2, 2]);
    }

    #[test]
    fn test_maxpool3d_forward_computation() {
        let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Simple 2x2x2 input with known values
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ];
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2, 2],
        )
        .unwrap();
        let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Expected: max of all 8 values = 8.0
        assert_eq!(output.shape().dims(), &[1, 1, 1, 1, 1]);
        assert_eq!(output.as_slice()[0].get(), 8.0);
    }

    #[test]
    fn test_maxpool3d_with_stride() {
        let pool = MaxPool3d::new((2, 2, 2), Some((1, 1, 1)), (0, 0, 0));

        // Input: [1, 1, 3, 3, 3]
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3, 3]).unwrap();
        let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output: (3 - 2) / 1 + 1 = 2
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_maxpool3d_batch_processing() {
        let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Input: [batch_size=4, channels=2, depth=4, height=4, width=4]
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[4, 2, 4, 4, 4]).unwrap();
        let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [4, 2, 2, 2, 2]
        assert_eq!(output.shape().dims(), &[4, 2, 2, 2, 2]);
    }

    #[test]
    fn test_maxpool3d_video_classification() {
        let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Video input: [1, 64, 16, 112, 112] (16 frames, 64 channels, 112x112 resolution)
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 112, 112])
                .unwrap();
        let output = <MaxPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output: [1, 64, 8, 56, 56] (downsampled by 2x)
        assert_eq!(output.shape().dims(), &[1, 64, 8, 56, 56]);
    }

    #[test]
    fn test_avgpool3d_constructor() {
        let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
        assert_eq!(pool.kernel_size, (2, 2, 2));
        assert_eq!(pool.stride, Some((2, 2, 2)));
        assert_eq!(pool.padding, (0, 0, 0));
    }

    #[test]
    fn test_avgpool3d_forward_shape() {
        let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Input: [batch_size=1, channels=3, depth=4, height=4, width=4]
        let input_data: Vec<Float32> = (0..192).map(|i| Float32::new(i as f32)).collect();
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 3, 4, 4, 4],
        )
        .unwrap();

        let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [1, 3, 2, 2, 2] (downsampled by 2x in all dimensions)
        assert_eq!(output.shape().dims(), &[1, 3, 2, 2, 2]);
    }

    #[test]
    fn test_avgpool3d_forward_computation() {
        let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Simple 2x2x2 input with known values
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ];
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2, 2],
        )
        .unwrap();
        let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Expected: avg of all 8 values = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5
        assert_eq!(output.shape().dims(), &[1, 1, 1, 1, 1]);
        assert_eq!(output.as_slice()[0].get(), 4.5);
    }

    #[test]
    fn test_avgpool3d_with_stride() {
        let pool = AvgPool3d::new((2, 2, 2), Some((1, 1, 1)), (0, 0, 0));

        // Input: [1, 1, 3, 3, 3]
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3, 3]).unwrap();
        let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output: (3 - 2) / 1 + 1 = 2
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2, 2]);
        // All values should be 1.0 (average of 1.0s)
        assert_eq!(output.as_slice()[0].get(), 1.0);
    }

    #[test]
    fn test_avgpool3d_batch_processing() {
        let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Input: [batch_size=4, channels=2, depth=4, height=4, width=4]
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[4, 2, 4, 4, 4]).unwrap();
        let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output shape should be [4, 2, 2, 2, 2]
        assert_eq!(output.shape().dims(), &[4, 2, 2, 2, 2]);
    }

    #[test]
    fn test_avgpool3d_video_classification() {
        let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

        // Video input: [1, 64, 16, 112, 112] (16 frames, 64 channels, 112x112 resolution)
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 112, 112])
                .unwrap();
        let output = <AvgPool3d as Module<CpuBackend, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();

        // Output: [1, 64, 8, 56, 56] (downsampled by 2x)
        assert_eq!(output.shape().dims(), &[1, 64, 8, 56, 56]);
    }
}
