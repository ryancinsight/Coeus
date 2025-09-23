//! 2D Adaptive Average Pooling layer
//!
//! Applies adaptive average pooling to 2D input tensors.
//! Adaptively adjusts pooling regions to achieve specific output dimensions.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 2D Adaptive Average Pooling layer
///
/// Applies adaptive average pooling to 2D input tensors.
/// Adaptively adjusts pooling regions to achieve specific output dimensions.
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
    /// * `output_size` - The desired output size (height, width)
    pub fn new(output_size: (usize, usize)) -> Self {
        Self {
            output_height: output_size.0,
            output_width: output_size.1,
        }
    }
}

impl<T: FloatDtype> Module<T> for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(NNError::InvalidInput {
                message: "AdaptiveAvgPool2d requires 4D input (batch_size, channels, height, width)".to_string(),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];

        let mut output_data = Vec::with_capacity(batch_size * channels * self.output_height * self.output_width);

        for batch in 0..batch_size {
            for channel in 0..channels {
                for out_h in 0..self.output_height {
                    for out_w in 0..self.output_width {
                        let h_start = (out_h * input_height) / self.output_height;
                        let h_end = ((out_h + 1) * input_height) / self.output_height;
                        let w_start = (out_w * input_width) / self.output_width;
                        let w_end = ((out_w + 1) * input_width) / self.output_width;

                        let mut sum = T::zero();
                        let mut count = 0usize;

                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let idx = ((batch * channels + channel) * input_height + h) * input_width + w;
                                sum = sum + input.data()[idx];
                                count += 1;
                            }
                        }

                        output_data.push(sum / T::from(count as f64).unwrap());
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
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}
