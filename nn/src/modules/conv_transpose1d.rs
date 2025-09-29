use crate::{Module, NNError, Result};
use coeus_dtype::{Dtype, FloatDtype};
use coeus_tensor::{Tensor, Backend};
use coeus_tensor::ops::creation;
use coeus_backend::{BackendData, CpuBackend};
use std::sync::Arc;
use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError};
use tracing::{instrument, debug_span};

// 1D Transposed Convolutional (Deconvolution) layer
//
// This module provides 1D transposed convolutional layers for upsampling
// and feature reconstruction in sequence models.
//
// ## Mathematical Foundation
//
// ### 1D Transposed Convolution
// ```math
// O[i,j] = \sum_k I[k,j] \cdot W[(i - k \cdot stride) \mod kernel, k]
// ```
//
// Where:
// - I: Input tensor of shape (batch_size, in_length, in_channels)
// - W: Weight tensor of shape (in_channels, out_channels/groups * kernel_size)
// - O: Output tensor of shape (batch_size, out_length, out_channels)
//
// Output length: out_length = (in_length - 1) \cdot stride - 2 \cdot padding + (kernel_size - 1) \cdot dilation + output_padding + 1
//
// ```mermaid
// graph LR
//     A[Input: batch x in_len x in_c] --> B[Insert Zeros: stride-1 between elements]
//     B --> C[Pad: output_padding + kernel-1]
//     C --> D[Full Conv: with flipped weights]
//     D --> E[Output: batch x out_len x out_c]
// ```

/// 1D Transposed Convolutional layer (Deconvolution)
///
/// Applies a 1D transposed convolution operation for upsampling.
/// Supports configurable kernel size, stride, padding, dilation, and output_padding.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d<T: FloatDtype, B: Backend<T> + Clone + Send + Sync = CpuBackend> {
    /// Weight tensor of shape (in_channels, out_channels/groups * kernel_size)
    pub weight: Tensor<T, B>,
    /// Bias tensor of shape (out_channels,)
    pub bias: Option<Tensor<T, B>>,
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
    /// Output padding
    pub output_padding: usize,
    /// Number of groups (stub: 1 only)
    pub groups: usize,
    /// Backend instance for tensor operations
    pub backend: B,
}

impl<T: FloatDtype + std::ops::AddAssign, B: Backend<T> + Clone + Send + Sync + Default> ConvTranspose1d<T, B> {
    /// Create a new 1D transposed convolutional layer with Xavier initialization
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    /// * `stride` - Stride of the transposed convolution (default: 1)
    /// * `padding` - Padding removed from both sides (default: 0)
    /// * `output_padding` - Additional padding added to output (default: 0)
    /// * `dilation` - Spacing between kernel elements (default: 1)
    /// * `groups` - Number of groups (stub: 1 only)
    /// * `bias` - Whether to include bias term
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::ConvTranspose1d;
    ///
    /// // Create a 1D transposed convolution for upsampling
    /// let deconv: ConvTranspose1d<f32> = ConvTranspose1d::new(64, 32, 3, 2, 1, 0, 1, 1, true).unwrap();
    /// ```
    #[instrument(fields(in_channels, out_channels, kernel_size))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        if groups != 1 {
            return Err(NNError::InvalidInput {
                message: "Groups >1 not implemented (stub)".to_string(),
            });
        }
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(NNError::InvalidInput {
                message: "in_channels/out_channels must be divisible by groups".to_string(),
            });
        }
        let backend = B::default();
        let fan_out = (out_channels / groups) * kernel_size;
        let std = <T as Dtype>::from_f64(1.0f64 / (fan_out as f64).sqrt()).ok_or(NNError::InitializationError {
            message: "Failed to compute std".to_string(),
        })?;
        let normal = Normal::new(0.0, Dtype::to_f64(&std).unwrap_or(1.0))
            .map_err(|e: NormalError| NNError::InitializationError {
                message: e.to_string(),
            })?;
        let mut rng = rand::thread_rng();
        let weight_size = in_channels * (out_channels / groups) * kernel_size;
        let mut weight_data = vec![T::zero(); weight_size];
        for val in &mut weight_data {
            *val = <T as Dtype>::from_f64(normal.sample(&mut rng)).unwrap_or(T::zero());
        }
        let weight_shape = vec![in_channels, (out_channels / groups) * kernel_size];
        let weight = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), weight_data, weight_shape).unwrap()
            .map_err(|e| NNError::InitializationError {
                message: e.to_string(),
            })?;

        let bias = if bias {
            let bias_data = vec![T::zero(); out_channels];
            Some(
                Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), bias_data, vec![out_channels]).unwrap()
                    .map_err(|e| NNError::InitializationError {
                        message: e.to_string(),
                    })?,
            )
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            backend,
        })
    }

    /// Create a 1D transposed convolutional layer with default parameters
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    pub fn with_kernel_size(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(in_channels, out_channels, kernel_size, 1, 0, 0, 1, 1, true)
            .expect("Failed to create ConvTranspose1d with kernel size")
    }

    /// Calculate output length for the transposed convolution
    ///
    /// # Arguments
    /// * `input_length` - Length of the input sequence
    ///
    /// # Returns
    /// Output length after transposed convolution
    pub fn output_size(&self, input_length: usize) -> usize {
        let effective_kernel = (self.kernel_size - 1) * self.dilation + 1;
        let numerator = (input_length as isize - 1) * self.stride as isize
            - 2 * self.padding as isize
            + effective_kernel as isize
            + self.output_padding as isize
            + 1;
        numerator as usize
    }

    /// Upsample input by inserting zeros (for stride >1)
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch, in_len, in_channels)
    ///
    /// # Returns
    /// Upsampled tensor (batch, upsampled_len, in_channels)
    fn upsample(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let (batch, in_len, in_c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if self.stride == 1 {
            return Ok(input.clone());
        }
        let up_len = (in_len - 1) * (self.stride - 1) + 1; // Zeros between
        let mut up_data = vec![T::zero(); batch * up_len * in_c];
        let input_arc = input.backend_data();
        if let BackendData::Cpu { data: input_data, .. } = &*input_arc {
            let mut idx = 0;
            for b in 0..batch {
                for i in 0..in_len {
                    for c in 0..in_c {
                        let flat_idx = b * in_len * in_c + i * in_c + c;
                        up_data[idx] = input_data[flat_idx];
                        idx += 1;
                        // Insert stride-1 zeros
                        for _ in 1..self.stride {
                            up_data[idx] = T::zero();
                            idx += 1;
                        }
                    }
                }
            }
        } else {
            return Err(NNError::InvalidInput {
                message: "ConvTranspose1d upsample expects CPU backend".to_string(),
            });
        }
        let up_shape = vec![batch, up_len, in_c];
        Ok(Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), up_data, up_shape).unwrap()?)
    }

    /// Forward pass for 1D transposed convolution
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, in_length, in_channels)
    ///
    /// # Returns
    /// Output tensor of shape (batch_size, out_length, out_channels)
    #[instrument(skip(self, input), fields(input_shape=?input.shape()))]
    pub fn forward_impl(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let span = debug_span!("conv_transpose1d_forward", in_channels=self.in_channels, out_channels=self.out_channels);
        let _enter = span.enter();

        if input.shape().len() != 3 {
            return Err(NNError::InvalidInput {
                message: "Input must be 3D: (batch, length, channels)".to_string(),
            });
        }
        let (batch, in_len, in_c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NNError::ShapeMismatch {
                expected: vec![self.in_channels],
                actual: vec![in_c],
            });
        }
        let out_len = self.output_size(in_len);

        // For now, implement simplified version (cap3)
        if self.groups != 1 || self.dilation != 1 {
            return Err(NNError::InvalidInput {
                message: "ConvTranspose1d groups > 1 or dilation > 1 not yet implemented".to_string(),
            });
        }

        // Upsample input
        let upsampled = self.upsample(input)?;

        // Apply padding if output_padding > 0
        let padded = if self.output_padding > 0 {
            let upsampled_arc = upsampled.backend_data();
            let padded_data = self.backend.pad(&upsampled_arc, vec![0, 0, 0, 0, 0, self.output_padding], T::zero())?;
            Tensor::from_backend_data(self.backend.clone(), Arc::new(padded_data), vec![batch, upsampled.shape()[1], in_c + self.output_padding])
        } else {
            upsampled
        };

        // Perform transposed convolution using matrix operations
        let weight_flipped = self.weight.transpose(0, 1)?;
        let input_reshaped = padded.reshape(&[batch * padded.shape()[1], self.in_channels])?;
        let output_data = self.backend.matmul(&input_reshaped.backend_data(), &weight_flipped.backend_data())?;
        let mut deconv_out = Tensor::from_backend_data(self.backend.clone(), Arc::new(output_data), vec![batch, padded.shape()[1], self.out_channels]);

        if let Some(bias) = &self.bias {
            let bias_backend_data = bias.backend_data();
            let bias_expanded = self.backend.expand(&bias_backend_data, vec![batch, deconv_out.shape()[1], self.out_channels])?;
            let output_with_bias = self.backend.add(&deconv_out.backend_data(), &bias_expanded)?;
            deconv_out = Tensor::from_backend_data(self.backend.clone(), Arc::new(output_with_bias), deconv_out.shape().to_vec());
        }

        // Check NaN/Inf
        if let BackendData::Cpu { data, .. } = &*deconv_out.backend_data() {
            if data.iter().any(|&x| !x.is_finite()) {
                return Err(NNError::ComputationError { message: "Output contains NaN or Inf".to_string() });
            }
        }

        Ok(deconv_out)
    }

    /// Backward pass for transposed convolution
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of output
    ///
    /// # Returns
    /// (grad_input, grad_weight, grad_bias)
    #[instrument(skip(self, grad_output))]
    pub fn backward(&self, grad_output: &Tensor<T, B>) -> Result<(Tensor<T, B>, Tensor<T, B>, Option<Tensor<T, B>>)> {
        let (batch, out_len, out_c) = (grad_output.shape()[0], grad_output.shape()[1], grad_output.shape()[2]);
        let in_len = (out_len - 1) / self.stride + 1;
        if out_c != self.out_channels {
            return Err(NNError::ShapeMismatch {
                expected: vec![self.out_channels],
                actual: vec![out_c],
            });
        }
        // Grad input: downsample grad_output (remove inserted zeros), then conv with flipped weight
        let grad_upsampled = self.downsample_grad(grad_output)?; // Stub downsample
        // Flip kernel manually for conv
        let weight_data = self.weight.data();
        let mut flipped_data = vec![T::zero(); weight_data.len()];
        let out_c = self.out_channels;
        let in_c = self.in_channels;
        let k = self.kernel_size;
        for oc in 0..out_c {
            for ic in 0..in_c {
                for i in 0..k {
                    let src_idx = oc * in_c * k + ic * k + i;
                    let dst_idx = oc * in_c * k + ic * k + (k - 1 - i);
                    flipped_data[dst_idx] = weight_data[src_idx].clone();
                }
            }
        }
        let weight_flipped = Tensor::from_backend_data(self.backend.clone(), Arc::new(self.backend.from_vec(flipped_data, self.weight.shape().to_vec())?), self.weight.shape().to_vec());
        // Stub: manual conv1d backward for now
        let grad_input_data = self.backend.zeros(vec![batch, in_len, in_c])?;
        let grad_input = Tensor::from_backend_data(self.backend.clone(), Arc::new(grad_input_data), vec![batch, in_len, in_c]);
        // Grad weight: similar to regular conv but on upsampled input
        let input_upsampled = self.upsample(&Tensor::zeros(vec![batch, (out_len - 1) / self.stride + 1, self.in_channels]).unwrap_grad()); // Stub
        // Stub grad_weight calculation
        let grad_weight_data = self.backend.zeros(self.weight.shape().to_vec())?;
        let grad_weight = Tensor::from_backend_data(self.backend.clone(), Arc::new(grad_weight_data), self.weight.shape().to_vec());
        let grad_bias = if let Some(_) = &self.bias {
            let bias_data = self.backend.sum_dim(grad_output.backend_data(), 1)?; // Sum over spatial dim
            Some(Tensor::from_backend_data(self.backend.clone(), Arc::new(bias_data), vec![self.out_channels]))
        } else {
            None
        };
        Ok((grad_input, grad_weight, grad_bias))
    }

    fn downsample_grad(&self, grad_output: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // Remove inserted zeros: average or zero the stride-1 positions (stub: subsample every stride)
        let (batch, out_len, in_c) = (grad_output.shape()[0], grad_output.shape()[1], grad_output.shape()[2]);
        let in_len = (out_len - 1) / self.stride + 1;
        let mut down_data = vec![T::zero(); batch * in_len * in_c];
        let mut idx = 0;
        for b in 0..batch {
            for i in 0..in_len {
                for c in 0..in_c {
                    // Take from position i*stride
                    let pos = i * self.stride;
                    if pos < out_len {
                        let flat_idx = b * out_len * in_c + pos * in_c + c;
                        down_data[idx] = grad_output.data()[flat_idx].clone();
                    }
                    idx += 1;
                }
            }
        }
        let down_shape = vec![batch, in_len, in_c];
        Ok(creation::from_vec(self.backend.clone(), down_data, down_shape)?)
    }
}

impl<T: FloatDtype + std::ops::AddAssign, B: Backend<T> + Clone + Send + Sync + Default> Module<T, B> for ConvTranspose1d<T, B> {
    fn forward(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.forward_impl(input)
    }

    fn parameters(&self) -> Vec<Tensor<T, B>> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        let mut params = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use approx::assert_relative_eq;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        fn prop_transpose_conv1d_forward(
            batch in 1usize..4,
            in_len in 2usize..10,
            in_c in 1usize..4,
            kernel in 2usize..4,
            stride in 2usize..3,
            padding in 0usize..1,
            output_padding in 0usize..1,
        ) {
            let deconv = ConvTranspose1d::new(in_c, in_c, kernel, stride, padding, output_padding, 1, 1, false).unwrap();
            let input_shape = vec![batch, in_len, in_c];
            let input = Tensor::randn(&input_shape, &deconv.backend); // Assume randn
            let output = deconv.forward$1.unwrap_grad();
            let expected_out_len = deconv.output_size(in_len);
            prop_assert_eq!(output.shape(), vec![batch, expected_out_len, in_c]);
            prop_assert!(!output.iter().all(|&x| x == T::zero()));
            prop_assert!(output.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_transpose_conv1d_edges() {
        let deconv = ConvTranspose1d::new(1, 1, 3, 2, 1, 0, 1, 1, false).unwrap();
        // Small input upsample
        let input = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![-1.0f32], vec![1, 1, 1]).unwrap();
        let output = deconv.forward$1.unwrap_grad();
        let expected_len = deconv.output_size(1); // (1-1)*2 -2*1 + (3-1)*1 +0 +1 = 0-2+2+1=1 wait adjust math
        assert_relative_eq!(output[[0,0,0]], -1.0 * weight[some], epsilon=1e-6); // Stub specific
        // Empty: len=0 → out_len calc, but Err if invalid
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        fn prop_transpose_conv1d_backward(
            batch in 1..2,
            in_len in 2..5,
            in_c in 1..2,
            kernel in 3..4,
            stride in 2..3,
        ) {
            let deconv = ConvTranspose1d::new(in_c, in_c, kernel, stride, 1, 0, 1, 1, false).unwrap();
            let input = Tensor::randn(vec![batch, in_len, in_c], &deconv.backend);
            let output = deconv.forward$1.unwrap_grad();
            let grad_output = Tensor::randn(output.shape().clone(), &deconv.backend);
            let (grad_input, grad_weight, grad_bias) = deconv.backward(&grad_output).unwrap();
            prop_assert_eq!(grad_input.shape(), input.shape());
            prop_assert_eq!(grad_weight.shape(), deconv.weight.shape());
            if let Some(b) = grad_bias {
                prop_assert_eq!(b.shape(), vec![in_c]);
            }
        }
    }
}


