use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 2D Average Pooling layer.
///
/// Applies a 2D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
///
/// Input shape: [N, C, H_in, W_in]
/// Output shape: [N, C, H_out, W_out]
///
/// where:
/// - H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
/// - W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create AvgPool2d with 2x2 kernel, stride 2
/// let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: [2, 64, 16, 16] (downsampled by 2x)
/// let output = <AvgPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (height, width)
    /// * `stride` - Stride (height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (height, width)
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let h_out = (input_h + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let w_out = (input_w + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        (h_out, w_out)
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for AvgPool2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 4, "Input must be 4D [N, C, H_in, W_in]");

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let (output_h, output_w) = self.output_size(input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        let kernel_area = T::from((self.kernel_size.0 * self.kernel_size.1) as f64).unwrap();

        for n in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        let mut sum = T::zero();

                        // Compute average in pooling window
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h_in = out_h * stride.0 + kh;
                                let w_in = out_w * stride.1 + kw;

                                // Handle padding (treat as 0 for average pooling)
                                if h_in >= self.padding.0
                                    && h_in < input_h + self.padding.0
                                    && w_in >= self.padding.1
                                    && w_in < input_w + self.padding.1
                                {
                                    let h_actual = h_in - self.padding.0;
                                    let w_actual = w_in - self.padding.1;

                                    if h_actual < input_h && w_actual < input_w {
                                        let input_idx = ((n * channels + c) * input_h + h_actual)
                                            * input_w
                                            + w_actual;
                                        sum = sum + input_data[input_idx];
                                    }
                                }
                            }
                        }

                        output_data.push(sum / kernel_area);
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, channels, output_h, output_w])
            .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        vec![] // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior
    }

    fn name(&self) -> &str {
        "AvgPool2d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
