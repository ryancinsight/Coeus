//! 1D Convolutional neural network layers
//!
//! This module provides 1D convolutional layers for sequence processing
//! and temporal feature extraction.
//!
//! ## Mathematical Foundation
//!
//! ### 1D Convolution
//! ```math
//! (O[i,j]) = Σₖ Σₘ (I[i, j+k, m] * W[m, k, j])
//!
//! Where:
//! - I: Input tensor of shape (batch_size, length, in_channels)
//! - W: Weight tensor of shape (out_channels, in_channels, kernel_size)
//! - O: Output tensor of shape (batch_size, out_length, out_channels)
//!
//! Output length: out_length = (length + 2*padding - kernel_size) / stride + 1
//! ```

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};
use rand::Rng;

/// 1D Convolutional layer
///
/// Applies a 1D convolution operation to input tensors.
/// Supports configurable kernel size, stride, padding, and dilation.
#[derive(Debug, Clone)]
pub struct Conv1d<T: FloatDtype> {
    /// Weight tensor of shape (out_channels, in_channels, kernel_size)
    pub weight: Tensor<T>,
    /// Bias tensor of shape (out_channels,)
    pub bias: Option<Tensor<T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Dilation
    pub dilation: usize,
}

impl<T: FloatDtype> Conv1d<T> {
    /// Create a new 1D convolutional layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding added to both sides (default: 0)
    /// * `dilation` - Spacing between kernel elements (default: 1)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Conv1d;
    ///
    /// // Create a 1D convolution with 32 input channels, 64 output channels, kernel size 3
    /// let conv: Conv1d<f32> = Conv1d::new(32, 64, 3, 1, 1, 1);
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        // Initialize weights with Kaiming initialization
        let weight_shape = vec![out_channels, in_channels, kernel_size];
        let fan_in = (in_channels * kernel_size) as f64;
        let std = (2.0 / fan_in).sqrt();

        let mut rng = rand::thread_rng();
        let mut weight_data = Vec::new();

        for _ in 0..weight_shape.iter().product::<usize>() {
            let value: f64 = rng.sample(rand_distr::Normal::new(0.0, std).unwrap());
            weight_data.push(T::from_f64(value).unwrap_or(T::zero()));
        }

        let weight = Tensor::from_vec(weight_data, weight_shape);

        // Initialize bias to zeros
        let bias = Some(Tensor::zeros(vec![out_channels]));

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Create a 1D convolutional layer with default stride, padding, and dilation
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Conv1d;
    ///
    /// let conv: Conv1d<f32> = Conv1d::with_kernel_size(32, 64, 3);
    /// ```
    pub fn with_kernel_size(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(in_channels, out_channels, kernel_size, 1, 0, 1)
    }

    /// Calculate output length for the convolution
    ///
    /// # Arguments
    /// * `input_length` - Length of the input sequence
    ///
    /// # Returns
    /// Output length after convolution
    pub fn output_size(&self, input_length: usize) -> usize {
        let kernel_size = (self.kernel_size - 1) * self.dilation + 1;
        let numerator = input_length + 2 * self.padding - kernel_size;
        numerator / self.stride + 1
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T> for Conv1d<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape validation
        if input.ndim() != 3 {
            return Err(NNError::ShapeMismatch {
                expected: vec![0, 0, self.in_channels],
                actual: input.shape().to_vec(),
            });
        }

        let batch_size = input.shape()[0];
        let input_length = input.shape()[1];
        let in_channels = input.shape()[2];

        if in_channels != self.in_channels {
            return Err(NNError::ShapeMismatch {
                expected: vec![batch_size, input_length, self.in_channels],
                actual: input.shape().to_vec(),
            });
        }

        let output_length = self.output_size(input_length);

        // Create output tensor
        let output_shape = vec![batch_size, output_length, self.out_channels];
        let mut output_data = Vec::new();

        // Naive implementation - in production, this should be optimized
        for b in 0..batch_size {
            for ol in 0..output_length {
                for oc in 0..self.out_channels {
                    let mut sum = T::zero();

                    for kc in 0..self.kernel_size {
                        let input_pos = (ol * self.stride + kc * self.dilation) as isize
                            - self.padding as isize;
                        if input_pos >= 0 && (input_pos as usize) < input_length {
                            for ic in 0..self.in_channels {
                                // Get input value: input[b, input_pos, ic]
                                let input_idx = b * input_length * in_channels
                                    + (input_pos as usize) * in_channels
                                    + ic;
                                let input_val = input.data()[input_idx];

                                // Get weight value: weight[oc, ic, kc]
                                let weight_idx = oc * in_channels * self.kernel_size
                                    + ic * self.kernel_size
                                    + kc;
                                let weight_val = self.weight.data()[weight_idx];

                                sum = sum + input_val * weight_val;
                            }
                        }
                    }

                    // Add bias
                    if let Some(ref bias) = self.bias {
                        sum = sum + bias.data()[oc];
                    }

                    output_data.push(sum);
                }
            }
        }

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
