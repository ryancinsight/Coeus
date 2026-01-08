use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 3D Average Pooling layer.
///
/// Applies a 3D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
/// Essential for video processing and 3D medical imaging.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)` where N is batch size, C is channels
/// - Output: `(N, C, D_out, H_out, W_out)` where:
///   - D_out = floor((D_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
///   - H_out = floor((H_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///   - W_out = floor((W_in + 2*padding[2] - kernel_size[2]) / stride[2] + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool3d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create AvgPool3d with 2x2x2 kernel, stride 2
/// let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
///
/// // Input: [batch_size=1, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: [1, 64, 8, 16, 16] (downsampled by 2x in all dimensions)
/// let output = <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 8, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize, usize)>,
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
}

impl AvgPool3d {
    /// Create a new AvgPool3d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (depth, height, width)
    /// * `stride` - Stride (depth, height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (depth, height, width)
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0 && s.2 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, input_d: usize, input_h: usize, input_w: usize) -> (usize, usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let d_out = (input_d + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let h_out = (input_h + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        let w_out = (input_w + 2 * self.padding.2 - self.kernel_size.2) / stride.2 + 1;
        (d_out, h_out, w_out)
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for AvgPool3d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, D_in, H_in, W_in]
        let input_shape = input.shape().dims();
        assert_eq!(
            input_shape.len(),
            5,
            "Input must be 5D [N, C, D_in, H_in, W_in]"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        let (output_d, output_h, output_w) = self.output_size(input_d, input_h, input_w);
        let stride = self.stride.unwrap_or(self.kernel_size);

        let input_data = input.as_slice();
        let mut output_data =
            Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);

        for n in 0..batch_size {
            for c in 0..channels {
                for out_d in 0..output_d {
                    for out_h in 0..output_h {
                        for out_w in 0..output_w {
                            let mut sum = T::zero();
                            let mut count = 0;

                            // Compute average in pooling window
                            for kd in 0..self.kernel_size.0 {
                                for kh in 0..self.kernel_size.1 {
                                    for kw in 0..self.kernel_size.2 {
                                        let d_in = out_d * stride.0 + kd;
                                        let h_in = out_h * stride.1 + kh;
                                        let w_in = out_w * stride.2 + kw;

                                        // Handle padding (treat as 0 for average pooling)
                                        if d_in >= self.padding.0
                                            && d_in < input_d + self.padding.0
                                            && h_in >= self.padding.1
                                            && h_in < input_h + self.padding.1
                                            && w_in >= self.padding.2
                                            && w_in < input_w + self.padding.2
                                        {
                                            let d_actual = d_in - self.padding.0;
                                            let h_actual = h_in - self.padding.1;
                                            let w_actual = w_in - self.padding.2;

                                            if d_actual < input_d
                                                && h_actual < input_h
                                                && w_actual < input_w
                                            {
                                                let input_idx = (((n * channels + c) * input_d
                                                    + d_actual)
                                                    * input_h
                                                    + h_actual)
                                                    * input_w
                                                    + w_actual;
                                                sum = sum + input_data[input_idx];
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                            }

                            let avg = if count > 0 {
                                sum / T::from(count).unwrap()
                            } else {
                                T::zero()
                            };
                            output_data.push(avg);
                        }
                    }
                }
            }
        }

        Tensor::from_vec(
            output_data,
            &[batch_size, channels, output_d, output_h, output_w],
        )
        .map_err(Into::into)
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
        "AvgPool3d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
