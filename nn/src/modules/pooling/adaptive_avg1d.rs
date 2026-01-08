use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

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
/// use nn::{AdaptiveAvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool1d::new(10); // Output length = 10
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
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
    for AdaptiveAvgPool1d
{
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
        "AdaptiveAvgPool1d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
