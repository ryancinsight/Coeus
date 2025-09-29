//! 2D Average Pooling layer
//!
//! Applies 2D average pooling operation to input tensors.
//! Reduces spatial dimensions by taking the average value in each kernel window.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};

/// 2D Average Pooling layer
///
/// Applies 2D average pooling operation to input tensors.
/// Reduces spatial dimensions by taking the average value in each kernel window.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in height dimension
    pub stride_height: usize,
    /// Stride in width dimension
    pub stride_width: usize,
    /// Padding in height dimension
    pub padding_height: usize,
    /// Padding in width dimension
    pub padding_width: usize,
    /// Whether to count padding in divisor for average
    pub count_include_pad: bool,
    /// Custom divisor (if None, uses kernel size)
    pub divisor_override: Option<usize>,
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer
    ///
    /// # Arguments
    /// * `kernel_height` - Height of the pooling kernel
    /// * `kernel_width` - Width of the pooling kernel
    /// * `stride_height` - Stride in height dimension (if None, uses kernel_height)
    /// * `stride_width` - Stride in width dimension (if None, uses kernel_width)
    /// * `padding_height` - Padding in height dimension
    /// * `padding_width` - Padding in width dimension
    /// * `count_include_pad` - Whether to include padding in average calculation
    /// * `divisor_override` - Custom divisor override (if None, uses kernel size)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_height: usize,
        kernel_width: usize,
        stride_height: Option<usize>,
        stride_width: Option<usize>,
        padding_height: usize,
        padding_width: usize,
        count_include_pad: bool,
        divisor_override: Option<usize>,
    ) -> Self {
        let stride_h = stride_height.unwrap_or(kernel_height);
        let stride_w = stride_width.unwrap_or(kernel_width);

        Self {
            kernel_height,
            kernel_width,
            stride_height: stride_h,
            stride_width: stride_w,
            padding_height,
            padding_width,
            count_include_pad,
            divisor_override,
        }
    }

    /// Calculate output size for 2D pooling
    fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = ((input_height + 2 * self.padding_height - self.kernel_height)
            / self.stride_height)
            + 1;
        let out_width =
            ((input_width + 2 * self.padding_width - self.kernel_width) / self.stride_width) + 1;

        (out_height, out_width)
    }

    /// Forward pass for 2D average pooling
    fn avg_pool2d_forward<T: FloatDtype>(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        let batch_size = input.shape()[0];
        let input_height = input.shape()[1];
        let input_width = input.shape()[2];
        let channels = input.shape()[3];

        let (output_height, output_width) = self.output_size(input_height, input_width);
        let output_shape = vec![batch_size, output_height, output_width, channels];

        let mut output_data = vec![T::zero(); output_shape.iter().product()];

        // Perform average pooling
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for c in 0..channels {
                        let mut sum = T::zero();
                        let mut count = 0usize;

                        // Sum values in kernel window
                        for kh in 0..self.kernel_height {
                            for kw in 0..self.kernel_width {
                                let ih = oh * self.stride_height + kh;
                                let iw = ow * self.stride_width + kw;

                                if ih >= self.padding_height
                                    && ih < input_height + self.padding_height
                                    && iw >= self.padding_width
                                    && iw < input_width + self.padding_width
                                {
                                    let actual_ih = ih - self.padding_height;
                                    let actual_iw = iw - self.padding_width;

                                    if actual_ih < input_height && actual_iw < input_width {
                                        let input_idx = ((b * input_height + actual_ih)
                                            * input_width
                                            + actual_iw)
                                            * channels
                                            + c;
                                        sum = sum + input.data()[input_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let output_idx =
                            ((b * output_height + oh) * output_width + ow) * channels + c;

                        if count > 0 {
                            let divisor = match self.divisor_override {
                                Some(d) => T::from(d).unwrap(),
                                None => {
                                    if self.count_include_pad {
                                        T::from(self.kernel_height * self.kernel_width).unwrap()
                                    } else {
                                        T::from(count).unwrap()
                                    }
                                }
                            };
                            output_data[output_idx] = sum / divisor;
                        } else {
                            output_data[output_idx] = T::zero();
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap())
    }
}

impl<T: FloatDtype> Module<T> for AvgPool2d {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        self.avg_pool2d_forward(input)
            .map_err(|e| NNError::InvalidInput {
                message: format!("AvgPool2d forward pass failed: {:?}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}


