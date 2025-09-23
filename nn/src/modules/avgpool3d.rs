//! 3D Average Pooling layer
//!
//! Applies 3D average pooling operation to input tensors.
//! Reduces spatial and depth dimensions by taking the average value in each kernel window.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 3D Average Pooling layer
///
/// Applies 3D average pooling operation to input tensors.
/// Reduces spatial and depth dimensions by taking the average value in each kernel window.
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    /// Kernel depth
    pub kernel_depth: usize,
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in depth dimension
    pub stride_depth: usize,
    /// Stride in height dimension
    pub stride_height: usize,
    /// Stride in width dimension
    pub stride_width: usize,
    /// Padding in depth dimension
    pub padding_depth: usize,
    /// Padding in height dimension
    pub padding_height: usize,
    /// Padding in width dimension
    pub padding_width: usize,
}

impl AvgPool3d {
    /// Create a new AvgPool3d layer
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling kernel (depth, height, width)
    /// * `stride` - Stride of the pooling operation (if None, uses kernel_size)
    /// * `padding` - Padding for the pooling operation
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: usize,
    ) -> Self {
        let (kernel_depth, kernel_height, kernel_width) = kernel_size;
        let (stride_depth, stride_height, stride_width) = stride.unwrap_or(kernel_size);

        Self {
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth: padding,
            padding_height: padding,
            padding_width: padding,
        }
    }

    /// Calculate output size for 3D pooling
    fn output_size(&self, input_depth: usize, input_height: usize, input_width: usize) -> (usize, usize, usize) {
        let out_depth = ((input_depth + 2 * self.padding_depth - self.kernel_depth) / self.stride_depth) + 1;
        let out_height = ((input_height + 2 * self.padding_height - self.kernel_height) / self.stride_height) + 1;
        let out_width = ((input_width + 2 * self.padding_width - self.kernel_width) / self.stride_width) + 1;

        (out_depth, out_height, out_width)
    }
}

impl<T: FloatDtype> Module<T> for AvgPool3d {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(NNError::InvalidInput {
                message: format!(
                    "AvgPool3d expects 5D input (batch_size, channels, depth, height, width), got {}D",
                    input.ndim()
                ),
            });
        }

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_depth = input.shape()[2];
        let input_height = input.shape()[3];
        let input_width = input.shape()[4];

        let (output_depth, output_height, output_width) = self.output_size(input_depth, input_height, input_width);
        let mut output_data = Vec::with_capacity(batch_size * channels * output_depth * output_height * output_width);

        for b in 0..batch_size {
            for c in 0..channels {
                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let mut sum = T::zero();
                            let mut count = 0usize;

                            for kd in 0..self.kernel_depth {
                                for kh in 0..self.kernel_height {
                                    for kw in 0..self.kernel_width {
                                        let id = od * self.stride_depth + kd;
                                        let ih = oh * self.stride_height + kh;
                                        let iw = ow * self.stride_width + kw;

                                        if id >= self.padding_depth && id < input_depth + self.padding_depth
                                            && ih >= self.padding_height && ih < input_height + self.padding_height
                                            && iw >= self.padding_width && iw < input_width + self.padding_width
                                        {
                                            let actual_id = id - self.padding_depth;
                                            let actual_ih = ih - self.padding_height;
                                            let actual_iw = iw - self.padding_width;

                                            if actual_id < input_depth && actual_ih < input_height && actual_iw < input_width {
                                                let idx = (((b * channels + c) * input_depth + actual_id)
                                                    * input_height + actual_ih)
                                                    * input_width + actual_iw;
                                                sum = sum + input.data()[idx];
                                                count += 1;
                                            }
                                        }
                                    }
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
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, channels, output_depth, output_height, output_width],
        ))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}
