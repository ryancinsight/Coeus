use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

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
/// use nn::{AdaptiveAvgPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 224, 224]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
/// ```
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
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Compute adaptive pooling parameters for a given dimension.
    ///
    /// Returns (start_index, end_index) for the pooling window.
    pub(crate) fn compute_adaptive_params(
        input_size: usize,
        output_size: usize,
        output_idx: usize,
    ) -> (usize, usize) {
        let start = (output_idx * input_size) / output_size;
        let end = ((output_idx + 1) * input_size) / output_size;
        (start, end)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T>
    for AdaptiveAvgPool2d
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
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
        let output_shape = if input_shape.len() == 3usize {
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

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
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

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
