//! Pooling layers for neural networks
//!
//! This module provides max pooling and average pooling operations
//! commonly used in convolutional neural networks for dimensionality reduction.
//!
//! ## Mathematical Foundation
//!
//! ### Max Pooling
//! ```math
//! O[i,j,k] = max_{u,v} I[i*stride+u, j*stride+v, k]
//! ```
//!
//! ### Average Pooling
//! ```math
//! O[i,j,k] = (1/(kernel_h * kernel_w)) * Σ_{u,v} I[i*stride+u, j*stride+v, k]
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Pooling](https://www.deeplearningbook.org/contents/convnets.html)
//! - [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor};

/// 2D Max Pooling layer
///
/// Applies 2D max pooling operation to input tensors.
/// Reduces spatial dimensions by taking the maximum value in each kernel window.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in height dimension
    pub stride_height: usize,
    /// Stride in width dimension
    pub stride_width: usize,
    /// Padding in height dimension
    pub padding_height: usize,
    /// Padding in width dimension
    pub padding_width: usize,
    /// Dilation in height dimension
    pub dilation_height: usize,
    /// Dilation in width dimension
    pub dilation_width: usize,
    /// Whether to return indices of max values (for MaxUnpool2d)
    pub return_indices: bool,
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer
    ///
    /// # Arguments
    /// * `kernel_height` - Height of the pooling kernel
    /// * `kernel_width` - Width of the pooling kernel
    /// * `stride_height` - Stride in height dimension (if None, uses kernel_height)
    /// * `stride_width` - Stride in width dimension (if None, uses kernel_width)
    /// * `padding_height` - Padding in height dimension
    /// * `padding_width` - Padding in width dimension
    /// * `dilation_height` - Dilation in height dimension
    /// * `dilation_width` - Dilation in width dimension
    /// * `return_indices` - Whether to return indices of max values
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_height: usize,
        kernel_width: usize,
        stride_height: Option<usize>,
        stride_width: Option<usize>,
        padding_height: usize,
        padding_width: usize,
        dilation_height: usize,
        dilation_width: usize,
        return_indices: bool,
    ) -> Self {
        let stride_h = stride_height.unwrap_or(kernel_height);
        let stride_w = stride_width.unwrap_or(kernel_width);

        Self {
            kernel_height,
            kernel_width,
            stride_height: stride_h,
            stride_width: stride_w,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            return_indices,
        }
    }

    /// Calculate output size for 2D pooling
    fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = ((input_height + 2 * self.padding_height
            - self.dilation_height * (self.kernel_height - 1)
            - 1)
            / self.stride_height)
            + 1;
        let out_width = ((input_width + 2 * self.padding_width
            - self.dilation_width * (self.kernel_width - 1)
            - 1)
            / self.stride_width)
            + 1;

        (out_height, out_width)
    }

    /// Forward pass for 2D max pooling
    fn max_pool2d_forward<T: FloatDtype>(
        &self,
        input: &Tensor<T>,
    ) -> Result<(Tensor<T>, Option<Tensor<i32>>), crate::NNError> {
        let batch_size = input.shape()[0];
        let input_height = input.shape()[1];
        let input_width = input.shape()[2];
        let channels = input.shape()[3];

        let (output_height, output_width) = self.output_size(input_height, input_width);
        let output_shape = vec![batch_size, output_height, output_width, channels];

        let mut output_data = vec![T::zero(); output_shape.iter().product()];
        let mut indices_data = if self.return_indices {
            Some(vec![0i32; output_shape.iter().product()])
        } else {
            None
        };

        // Perform max pooling
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for c in 0..channels {
                        let mut max_val = T::neg_infinity();
                        let mut max_idx = 0i32;

                        // Find max value in kernel window
                        for kh in 0..self.kernel_height {
                            for kw in 0..self.kernel_width {
                                let ih = oh * self.stride_height + kh * self.dilation_height;
                                let iw = ow * self.stride_width + kw * self.dilation_width;

                                if ih >= self.padding_height
                                    && ih < input_height + self.padding_height
                                    && iw >= self.padding_width
                                    && iw < input_width + self.padding_width
                                {
                                    let actual_ih = ih - self.padding_height;
                                    let actual_iw = iw - self.padding_width;

                                    if actual_ih < input_height && actual_iw < input_width {
                                        let input_idx = ((b * input_height + actual_ih)
                                            * input_width
                                            + actual_iw)
                                            * channels
                                            + c;
                                        let input_val = input.data()[input_idx];

                                        if input_val > max_val {
                                            max_val = input_val;
                                            max_idx = (kh * self.kernel_width + kw) as i32;
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx =
                            ((b * output_height + oh) * output_width + ow) * channels + c;
                        output_data[output_idx] = max_val;

                        if let Some(ref mut indices) = indices_data {
                            indices[output_idx] = max_idx;
                        }
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, output_shape.clone());
        let indices = indices_data.map(|data| Tensor::from_vec(data, output_shape));

        Ok((output, indices))
    }
}

impl Module<f32> for MaxPool2d {
    fn forward(&self, input: &Tensor<f32>) -> crate::Result<Tensor<f32>> {
        self.max_pool2d_forward(input)
            .map(|(output, _)| output)
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("MaxPool2d forward pass failed: {:?}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![] // MaxPool2d has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // MaxPool2d has no learnable parameters
    }
}

/// 2D Average Pooling layer
///
/// Applies 2D average pooling operation to input tensors.
/// Reduces spatial dimensions by taking the average value in each kernel window.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in height dimension
    pub stride_height: usize,
    /// Stride in width dimension
    pub stride_width: usize,
    /// Padding in height dimension
    pub padding_height: usize,
    /// Padding in width dimension
    pub padding_width: usize,
    /// Whether to count padding in divisor for average
    pub count_include_pad: bool,
    /// Custom divisor (if None, uses kernel size)
    pub divisor_override: Option<usize>,
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer
    ///
    /// # Arguments
    /// * `kernel_height` - Height of the pooling kernel
    /// * `kernel_width` - Width of the pooling kernel
    /// * `stride_height` - Stride in height dimension (if None, uses kernel_height)
    /// * `stride_width` - Stride in width dimension (if None, uses kernel_width)
    /// * `padding_height` - Padding in height dimension
    /// * `padding_width` - Padding in width dimension
    /// * `count_include_pad` - Whether to include padding in average calculation
    /// * `divisor_override` - Custom divisor override (if None, uses kernel size)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_height: usize,
        kernel_width: usize,
        stride_height: Option<usize>,
        stride_width: Option<usize>,
        padding_height: usize,
        padding_width: usize,
        count_include_pad: bool,
        divisor_override: Option<usize>,
    ) -> Self {
        let stride_h = stride_height.unwrap_or(kernel_height);
        let stride_w = stride_width.unwrap_or(kernel_width);

        Self {
            kernel_height,
            kernel_width,
            stride_height: stride_h,
            stride_width: stride_w,
            padding_height,
            padding_width,
            count_include_pad,
            divisor_override,
        }
    }

    /// Calculate output size for 2D pooling
    fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = ((input_height + 2 * self.padding_height - self.kernel_height)
            / self.stride_height)
            + 1;
        let out_width =
            ((input_width + 2 * self.padding_width - self.kernel_width) / self.stride_width) + 1;

        (out_height, out_width)
    }

    /// Forward pass for 2D average pooling
    fn avg_pool2d_forward<T: FloatDtype>(
        &self,
        input: &Tensor<T>,
    ) -> Result<Tensor<T>, crate::NNError> {
        let batch_size = input.shape()[0];
        let input_height = input.shape()[1];
        let input_width = input.shape()[2];
        let channels = input.shape()[3];

        let (output_height, output_width) = self.output_size(input_height, input_width);
        let output_shape = vec![batch_size, output_height, output_width, channels];

        let mut output_data = vec![T::zero(); output_shape.iter().product()];

        // Perform average pooling
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for c in 0..channels {
                        let mut sum = T::zero();
                        let mut count = 0usize;

                        // Sum values in kernel window
                        for kh in 0..self.kernel_height {
                            for kw in 0..self.kernel_width {
                                let ih = oh * self.stride_height + kh;
                                let iw = ow * self.stride_width + kw;

                                if ih >= self.padding_height
                                    && ih < input_height + self.padding_height
                                    && iw >= self.padding_width
                                    && iw < input_width + self.padding_width
                                {
                                    let actual_ih = ih - self.padding_height;
                                    let actual_iw = iw - self.padding_width;

                                    if actual_ih < input_height && actual_iw < input_width {
                                        let input_idx = ((b * input_height + actual_ih)
                                            * input_width
                                            + actual_iw)
                                            * channels
                                            + c;
                                        sum = sum + input.data()[input_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let output_idx =
                            ((b * output_height + oh) * output_width + ow) * channels + c;

                        if count > 0 {
                            let divisor = match self.divisor_override {
                                Some(d) => T::from(d).unwrap(),
                                None => {
                                    if self.count_include_pad {
                                        T::from(self.kernel_height * self.kernel_width).unwrap()
                                    } else {
                                        T::from(count).unwrap()
                                    }
                                }
                            };
                            output_data[output_idx] = sum / divisor;
                        } else {
                            output_data[output_idx] = T::zero();
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(output_data, output_shape))
    }
}

impl Module<f32> for AvgPool2d {
    fn forward(&self, input: &Tensor<f32>) -> crate::Result<Tensor<f32>> {
        self.avg_pool2d_forward(input)
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("AvgPool2d forward pass failed: {:?}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![] // AvgPool2d has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // AvgPool2d has no learnable parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d_forward() {
        let max_pool = MaxPool2d::new(
            2,       // kernel_height
            2,       // kernel_width
            Some(2), // stride_height
            Some(2), // stride_width
            0,       // padding_height
            0,       // padding_width
            1,       // dilation_height
            1,       // dilation_width
            false,   // return_indices
        );

        // Input: 1x4x4x1 (batch_size=1, height=4, width=4, channels=1)
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_vec(input_data, vec![1, 4, 4, 1]);

        let output = max_pool.forward(&input).unwrap();

        // With 2x2 kernel and stride 2 on 4x4 input, output should be 2x2
        assert_eq!(output.shape(), &[1, 2, 2, 1]);

        // Check some max values
        let output_data = output.data();
        assert_eq!(output_data[0], 6.0); // max of [1,2,5,6]
        assert_eq!(output_data[1], 8.0); // max of [3,4,7,8]
        assert_eq!(output_data[2], 14.0); // max of [9,10,13,14]
        assert_eq!(output_data[3], 16.0); // max of [11,12,15,16]
    }

    #[test]
    fn test_avg_pool2d_forward() {
        let avg_pool = AvgPool2d::new(
            2,       // kernel_height
            2,       // kernel_width
            Some(2), // stride_height
            Some(2), // stride_width
            0,       // padding_height
            0,       // padding_width
            true,    // count_include_pad
            None,    // divisor_override
        );

        // Input: 1x4x4x1 (batch_size=1, height=4, width=4, channels=1)
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_vec(input_data, vec![1, 4, 4, 1]);

        let output = avg_pool.forward(&input).unwrap();

        // With 2x2 kernel and stride 2 on 4x4 input, output should be 2x2
        assert_eq!(output.shape(), &[1, 2, 2, 1]);

        // Check some average values
        let output_data = output.data();
        assert_eq!(output_data[0], 3.5); // avg of [1,2,5,6] = (1+2+5+6)/4 = 14/4 = 3.5
        assert_eq!(output_data[1], 5.5); // avg of [3,4,7,8] = (3+4+7+8)/4 = 22/4 = 5.5
        assert_eq!(output_data[2], 11.5); // avg of [9,10,13,14] = (9+10+13+14)/4 = 46/4 = 11.5
        assert_eq!(output_data[3], 13.5); // avg of [11,12,15,16] = (11+12+15+16)/4 = 54/4 = 13.5
    }

    #[test]
    fn test_adaptive_avg_pool1d_basic() {
        let pool = AdaptiveAvgPool1d::new(2);
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 channels, length 3
        let input = Tensor::from_vec(input_data, vec![1, 2, 3]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(output.data().len(), 4);
    }

    #[test]
    fn test_adaptive_avg_pool2d_basic() {
        let pool = AdaptiveAvgPool2d::new(2, 2);
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]; // 1 channel, 4x4
        let input = Tensor::from_vec(input_data, vec![1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.data().len(), 4);
    }

    #[test]
    fn test_adaptive_max_pool1d_basic() {
        let pool = AdaptiveMaxPool1d::new(2);
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 channels, length 3
        let input = Tensor::from_vec(input_data, vec![1, 2, 3]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(output.data().len(), 4);
    }

    #[test]
    fn test_adaptive_max_pool2d_basic() {
        let pool = AdaptiveMaxPool2d::new(2, 2);
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]; // 1 channel, 4x4
        let input = Tensor::from_vec(input_data, vec![1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.data().len(), 4);
    }

    #[test]
    fn test_adaptive_pool_invalid_input_size() {
        let pool = AdaptiveAvgPool1d::new(10); // Output size larger than input
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);

        let result = pool.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_pool_invalid_dimensions() {
        let pool = AdaptiveAvgPool1d::new(2);
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]); // 2D instead of 3D

        let result = pool.forward(&input);
        assert!(result.is_err());
    }
}

/// 1D Adaptive Average Pooling layer
///
/// Applies 1D adaptive average pooling to reduce input to a fixed output size.
/// The kernel size is automatically determined to produce the desired output size.
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool1d {
    /// Target output size
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create a new AdaptiveAvgPool1d layer
    ///
    /// # Arguments
    /// * `output_size` - The desired output size
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveAvgPool1d;
    ///
    /// let pool = AdaptiveAvgPool1d::new(4);
    /// ```
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveAvgPool1d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(crate::NNError::InvalidInput {
                message: "AdaptiveAvgPool1d requires 3D input (batch_size, channels, length)"
                    .to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];

        if self.output_size > input_length {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Output size {} cannot be larger than input length {}",
                    self.output_size, input_length
                ),
            });
        }

        let mut output_data = Vec::with_capacity(batch_size * channels * self.output_size);

        for batch in 0..batch_size {
            for channel in 0..channels {
                // Calculate pooling regions for each output position
                for out_pos in 0..self.output_size {
                    let start_idx = (out_pos * input_length) / self.output_size;
                    let end_idx = ((out_pos + 1) * input_length) / self.output_size;
                    let kernel_size = end_idx - start_idx;

                    let mut sum = T::zero();
                    for i in start_idx..end_idx {
                        let idx = batch * channels * input_length + channel * input_length + i;
                        sum = sum + input.data()[idx];
                    }

                    let avg = sum / T::from(kernel_size as f64).unwrap();
                    output_data.push(avg);
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, channels, self.output_size],
        ))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

/// 2D Adaptive Average Pooling layer
///
/// Applies 2D adaptive average pooling to reduce input to a fixed output size.
/// The kernel size is automatically determined to produce the desired output size.
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    /// Target output height
    pub output_height: usize,
    /// Target output width
    pub output_width: usize,
}

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d layer
    ///
    /// # Arguments
    /// * `output_height` - The desired output height
    /// * `output_width` - The desired output width
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveAvgPool2d;
    ///
    /// let pool = AdaptiveAvgPool2d::new(4, 4);
    /// ```
    pub fn new(output_height: usize, output_width: usize) -> Self {
        Self {
            output_height,
            output_width,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(crate::NNError::InvalidInput {
                message:
                    "AdaptiveAvgPool2d requires 4D input (batch_size, channels, height, width)"
                        .to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];

        if self.output_height > input_height || self.output_width > input_width {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Output size {}x{} cannot be larger than input size {}x{}",
                    self.output_height, self.output_width, input_height, input_width
                ),
            });
        }

        let mut output_data =
            Vec::with_capacity(batch_size * channels * self.output_height * self.output_width);

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_h in 0..self.output_height {
                    // Calculate pooling regions for each output position
                    let h_start = (out_h * input_height) / self.output_height;
                    let h_end = ((out_h + 1) * input_height) / self.output_height;

                    for out_w in 0..self.output_width {
                        let w_start = (out_w * input_width) / self.output_width;
                        let w_end = ((out_w + 1) * input_width) / self.output_width;

                        let kernel_area = (h_end - h_start) * (w_end - w_start);

                        let mut sum = T::zero();
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let idx = batch * channels * input_height * input_width
                                    + channel * input_height * input_width
                                    + h * input_width
                                    + w;
                                sum = sum + input.data()[idx];
                            }
                        }

                        let avg = sum / T::from(kernel_area as f64).unwrap();
                        output_data.push(avg);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, channels, self.output_height, self.output_width],
        ))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

/// 1D Adaptive Max Pooling layer
///
/// Applies 1D adaptive max pooling to reduce input to a fixed output size.
/// The kernel size is automatically determined to produce the desired output size.
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool1d {
    /// Target output size
    pub output_size: usize,
    /// Whether to return indices of max values
    pub return_indices: bool,
}

impl AdaptiveMaxPool1d {
    /// Create a new AdaptiveMaxPool1d layer
    ///
    /// # Arguments
    /// * `output_size` - The desired output size
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool1d;
    ///
    /// let pool = AdaptiveMaxPool1d::new(4);
    /// ```
    pub fn new(output_size: usize) -> Self {
        Self {
            output_size,
            return_indices: false,
        }
    }

    /// Create a new AdaptiveMaxPool1d layer that returns indices
    ///
    /// # Arguments
    /// * `output_size` - The desired output size
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool1d;
    ///
    /// let pool = AdaptiveMaxPool1d::new_with_indices(4);
    /// ```
    pub fn new_with_indices(output_size: usize) -> Self {
        Self {
            output_size,
            return_indices: true,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveMaxPool1d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(crate::NNError::InvalidInput {
                message: "AdaptiveMaxPool1d requires 3D input (batch_size, channels, length)"
                    .to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];

        if self.output_size > input_length {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Output size {} cannot be larger than input length {}",
                    self.output_size, input_length
                ),
            });
        }

        let output_size = if self.return_indices {
            batch_size * channels * self.output_size * 2 // values + indices
        } else {
            batch_size * channels * self.output_size
        };

        let mut output_data = Vec::with_capacity(output_size);

        for batch in 0..batch_size {
            for channel in 0..channels {
                // Calculate pooling regions for each output position
                for out_pos in 0..self.output_size {
                    let start_idx = (out_pos * input_length) / self.output_size;
                    let end_idx = ((out_pos + 1) * input_length) / self.output_size;

                    let mut max_val = input.data()
                        [batch * channels * input_length + channel * input_length + start_idx];
                    let mut max_idx = start_idx;

                    for i in (start_idx + 1)..end_idx {
                        let idx = batch * channels * input_length + channel * input_length + i;
                        let val = input.data()[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = i;
                        }
                    }

                    output_data.push(max_val);
                    if self.return_indices {
                        output_data.push(T::from(max_idx as f64).unwrap());
                    }
                }
            }
        }

        let shape = if self.return_indices {
            vec![batch_size, channels, self.output_size, 2]
        } else {
            vec![batch_size, channels, self.output_size]
        };

        Ok(Tensor::from_vec(output_data, shape))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

/// 2D Adaptive Max Pooling layer
///
/// Applies 2D adaptive max pooling to reduce input to a fixed output size.
/// The kernel size is automatically determined to produce the desired output size.
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    /// Target output height
    pub output_height: usize,
    /// Target output width
    pub output_width: usize,
    /// Whether to return indices of max values
    pub return_indices: bool,
}

impl AdaptiveMaxPool2d {
    /// Create a new AdaptiveMaxPool2d layer
    ///
    /// # Arguments
    /// * `output_height` - The desired output height
    /// * `output_width` - The desired output width
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool2d;
    ///
    /// let pool = AdaptiveMaxPool2d::new(4, 4);
    /// ```
    pub fn new(output_height: usize, output_width: usize) -> Self {
        Self {
            output_height,
            output_width,
            return_indices: false,
        }
    }

    /// Create a new AdaptiveMaxPool2d layer that returns indices
    ///
    /// # Arguments
    /// * `output_height` - The desired output height
    /// * `output_width` - The desired output width
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool2d;
    ///
    /// let pool = AdaptiveMaxPool2d::new_with_indices(4, 4);
    /// ```
    pub fn new_with_indices(output_height: usize, output_width: usize) -> Self {
        Self {
            output_height,
            output_width,
            return_indices: true,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveMaxPool2d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(crate::NNError::InvalidInput {
                message:
                    "AdaptiveMaxPool2d requires 4D input (batch_size, channels, height, width)"
                        .to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];

        if self.output_height > input_height || self.output_width > input_width {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Output size {}x{} cannot be larger than input size {}x{}",
                    self.output_height, self.output_width, input_height, input_width
                ),
            });
        }

        let output_size = if self.return_indices {
            batch_size * channels * self.output_height * self.output_width * 2 // values + indices
        } else {
            batch_size * channels * self.output_height * self.output_width
        };

        let mut output_data = Vec::with_capacity(output_size);

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_h in 0..self.output_height {
                    // Calculate pooling regions for each output position
                    let h_start = (out_h * input_height) / self.output_height;
                    let h_end = ((out_h + 1) * input_height) / self.output_height;

                    for out_w in 0..self.output_width {
                        let w_start = (out_w * input_width) / self.output_width;
                        let w_end = ((out_w + 1) * input_width) / self.output_width;

                        let mut max_val =
                            input.data()[batch * channels * input_height * input_width
                                + channel * input_height * input_width
                                + h_start * input_width
                                + w_start];
                        let mut max_h = h_start;
                        let mut max_w = w_start;

                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let idx = batch * channels * input_height * input_width
                                    + channel * input_height * input_width
                                    + h * input_width
                                    + w;
                                let val = input.data()[idx];
                                if val > max_val {
                                    max_val = val;
                                    max_h = h;
                                    max_w = w;
                                }
                            }
                        }

                        output_data.push(max_val);
                        if self.return_indices {
                            // Store as flattened index for compatibility
                            let flat_idx = max_h * input_width + max_w;
                            output_data.push(T::from(flat_idx as f64).unwrap());
                        }
                    }
                }
            }
        }

        let shape = if self.return_indices {
            vec![
                batch_size,
                channels,
                self.output_height,
                self.output_width,
                2,
            ]
        } else {
            vec![batch_size, channels, self.output_height, self.output_width]
        };

        Ok(Tensor::from_vec(output_data, shape))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

/// 3D Adaptive Average Pooling layer
///
/// Resizes 5D input tensors to a specified output size using average pooling.
/// Commonly used for global average pooling in 3D convolutional networks.
///
/// # Arguments
/// * `output_size` - Desired output size as (depth, height, width)
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool3d {
    /// Target output depth
    pub output_depth: usize,
    /// Target output height
    pub output_height: usize,
    /// Target output width
    pub output_width: usize,
}

impl AdaptiveAvgPool3d {
    /// Create a new AdaptiveAvgPool3d layer
    ///
    /// # Arguments
    /// * `output_size` - Tuple of (depth, height, width) for output size
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveAvgPool3d;
    ///
    /// let pool = AdaptiveAvgPool3d::new((4, 4, 4));
    /// ```
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self {
            output_depth: output_size.0,
            output_height: output_size.1,
            output_width: output_size.2,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveAvgPool3d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(crate::NNError::InvalidInput {
                message: format!("AdaptiveAvgPool3d requires 5D input, got {}D", input.ndim()),
            });
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let input_depth = shape[2];
        let input_height = shape[3];
        let input_width = shape[4];

        let mut output_data = Vec::with_capacity(
            batch_size * channels * self.output_depth * self.output_height * self.output_width
        );

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_d in 0..self.output_depth {
                    for out_h in 0..self.output_height {
                        for out_w in 0..self.output_width {
                            // Calculate input region for this output element
                            let d_start = (out_d * input_depth) / self.output_depth;
                            let d_end = ((out_d + 1) * input_depth) / self.output_depth;
                            let h_start = (out_h * input_height) / self.output_height;
                            let h_end = ((out_h + 1) * input_height) / self.output_height;
                            let w_start = (out_w * input_width) / self.output_width;
                            let w_end = ((out_w + 1) * input_width) / self.output_width;

                            // Compute average of the region
                            let mut sum = T::zero();
                            let mut count = 0;

                            for d in d_start..d_end {
                                for h in h_start..h_end {
                                    for w in w_start..w_end {
                                        let idx = ((batch * channels + channel) * input_depth + d)
                                                * input_height * input_width
                                                + h * input_width + w;
                                        sum = sum + input.data()[idx];
                                        count += 1;
                                    }
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
        }

        let output_shape = vec![
            batch_size,
            channels,
            self.output_depth,
            self.output_height,
            self.output_width,
        ];

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

impl std::fmt::Display for AdaptiveAvgPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AdaptiveAvgPool3d(output_size=({}, {}, {}))",
            self.output_depth, self.output_height, self.output_width
        )
    }
}

/// 3D Adaptive Max Pooling layer
///
/// Resizes 5D input tensors to a specified output size using max pooling.
/// Commonly used for global max pooling in 3D convolutional networks.
///
/// # Arguments
/// * `output_size` - Desired output size as (depth, height, width)
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool3d {
    /// Target output depth
    pub output_depth: usize,
    /// Target output height
    pub output_height: usize,
    /// Target output width
    pub output_width: usize,
}

impl AdaptiveMaxPool3d {
    /// Create a new AdaptiveMaxPool3d layer
    ///
    /// # Arguments
    /// * `output_size` - Tuple of (depth, height, width) for output size
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::AdaptiveMaxPool3d;
    ///
    /// let pool = AdaptiveMaxPool3d::new((4, 4, 4));
    /// ```
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self {
            output_depth: output_size.0,
            output_height: output_size.1,
            output_width: output_size.2,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveMaxPool3d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(crate::NNError::InvalidInput {
                message: format!("AdaptiveMaxPool3d requires 5D input, got {}D", input.ndim()),
            });
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let input_depth = shape[2];
        let input_height = shape[3];
        let input_width = shape[4];

        let mut output_data = Vec::with_capacity(
            batch_size * channels * self.output_depth * self.output_height * self.output_width
        );

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_d in 0..self.output_depth {
                    for out_h in 0..self.output_height {
                        for out_w in 0..self.output_width {
                            // Calculate input region for this output element
                            let d_start = (out_d * input_depth) / self.output_depth;
                            let d_end = ((out_d + 1) * input_depth) / self.output_depth;
                            let h_start = (out_h * input_height) / self.output_height;
                            let h_end = ((out_h + 1) * input_height) / self.output_height;
                            let w_start = (out_w * input_width) / self.output_width;
                            let w_end = ((out_w + 1) * input_width) / self.output_width;

                            // Find maximum in the region
                            let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                            for d in d_start..d_end {
                                for h in h_start..h_end {
                                    for w in w_start..w_end {
                                        let idx = ((batch * channels + channel) * input_depth + d)
                                                * input_height * input_width
                                                + h * input_width + w;
                                        let val = input.data()[idx];
                                        if val > max_val {
                                            max_val = val;
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

        let output_shape = vec![
            batch_size,
            channels,
            self.output_depth,
            self.output_height,
            self.output_width,
        ];

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

impl std::fmt::Display for AdaptiveMaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AdaptiveMaxPool3d(output_size=({}, {}, {}))",
            self.output_depth, self.output_height, self.output_width
        )
    }
}
