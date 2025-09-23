//! 1D Adaptive Average Pooling layer
//!
//! Applies adaptive average pooling to 1D input tensors.
//! Adaptively adjusts pooling regions to achieve a specific output size.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 1D Adaptive Average Pooling layer
///
/// Applies adaptive average pooling to 1D input tensors.
/// Adaptively adjusts pooling regions to achieve a specific output size.
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
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveAvgPool1d {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(NNError::InvalidInput {
                message: "AdaptiveAvgPool1d requires 3D input (batch_size, channels, length)".to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];

        if self.output_size > input_length {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Output size {} cannot be larger than input length {}",
                    self.output_size, input_length
                ),
            });
        }

        let mut output_data = Vec::with_capacity(batch_size * channels * self.output_size);

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_pos in 0..self.output_size {
                    let start_idx = (out_pos * input_length) / self.output_size;
                    let end_idx = ((out_pos + 1) * input_length) / self.output_size;
                    let kernel_size = end_idx - start_idx;

                    let mut sum = T::zero();
                    for i in start_idx..end_idx {
                        let idx = batch * channels * input_length + channel * input_length + i;
                        sum = sum + input.data()[idx];
                    }

                    output_data.push(sum / T::from(kernel_size as f64).unwrap());
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, channels, self.output_size],
        ))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}
