//! 1D Max Pooling layer
//!
//! Applies 1D max pooling operation to input tensors.
//! Reduces temporal dimension by taking the maximum value in each kernel window.

use crate::{Module, NNError, Result};
use coeus_backend::CpuBackend;
use coeus_tensor::{FloatDtype, Tensor};

/// 1D Max Pooling layer
///
/// Applies 1D max pooling operation to input tensors.
/// Reduces temporal dimension by taking the maximum value in each kernel window.
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride of the pooling operation
    pub stride: usize,
    /// Padding for the pooling operation
    pub padding: usize,
}

impl MaxPool1d {
    /// Create a new MaxPool1d layer
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling kernel
    /// * `stride` - Stride of the pooling operation (if None, uses kernel_size)
    /// * `padding` - Padding for the pooling operation
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Calculate output size for 1D pooling
    fn output_size(&self, input_length: usize) -> usize {
        ((input_length + 2 * self.padding - self.kernel_size) / self.stride) + 1
    }
}

impl<T: FloatDtype> Module<T> for MaxPool1d {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        if input.ndim() != 3 {
            return Err(NNError::InvalidInput {
                message: format!(
                    "MaxPool1d expects 3D input (batch_size, channels, length), got {}D",
                    input.ndim()
                ),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];
        let output_length = self.output_size(input_length);

        let mut output_data = Vec::with_capacity(batch_size * channels * output_length);

        for b in 0..batch_size {
            for c in 0..channels {
                for i in 0..output_length {
                    let start = i * self.stride;
                    let end = (start + self.kernel_size).min(input_length);
                    let mut max_val = T::neg_infinity();

                    for j in start..end {
                        let idx = (b * channels + c) * input_length + j;
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
            vec![batch_size, channels, output_length],
        ).unwrap())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}



