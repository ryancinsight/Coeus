//! 1D Adaptive Max Pooling layer
//!
//! Applies adaptive max pooling to 1D input tensors.
//! Adaptively adjusts pooling regions to achieve a specific output size.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};

/// 1D Adaptive Max Pooling layer
///
/// Applies adaptive max pooling to 1D input tensors.
/// Adaptively adjusts pooling regions to achieve a specific output size.
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool1d {
    /// Target output size
    pub output_size: usize,
}

impl AdaptiveMaxPool1d {
    /// Create a new AdaptiveMaxPool1d layer
    ///
    /// # Arguments
    /// * `output_size` - The desired output size
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveMaxPool1d {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        if input.ndim() != 3 {
            return Err(NNError::InvalidInput {
                message: "AdaptiveMaxPool1d requires 3D input (batch_size, channels, length)".to_string(),
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

                    let mut max_val = T::neg_infinity();
                    for i in start_idx..end_idx {
                        let idx = batch * channels * input_length + channel * input_length + i;
                        if input.data()[idx] > max_val {
                            max_val = input.data()[idx];
                        }
                    }

                    output_data.push(max_val);
                }
            }
        }

        Ok(Tensor::from_vec(
            CpuBackend::default(),
            output_data,
            vec![batch_size, channels, self.output_size],
        ).unwrap())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}


