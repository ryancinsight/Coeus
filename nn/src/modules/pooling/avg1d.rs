use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

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
/// use nn::{AvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AvgPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
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

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for AvgPool1d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
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

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
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

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
