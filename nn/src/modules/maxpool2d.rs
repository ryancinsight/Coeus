//! 2D Max Pooling layer
//!
//! Applies 2D max pooling operation to input tensors.
//! Reduces spatial dimensions by taking the maximum value in each kernel window.

use crate::{Module, NNError, Result};
use coeus_backend::CpuBackend;
use coeus_tensor::{FloatDtype, Tensor};

/// 2D Max Pooling layer
///
/// Applies 2D max pooling operation to input tensors.
/// Reduces spatial dimensions by taking the maximum value in each kernel window.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
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
    /// Dilation in height dimension
    pub dilation_height: usize,
    /// Dilation in width dimension
    pub dilation_width: usize,
    /// Whether to return indices of max values (for MaxUnpool2d)
    pub return_indices: bool,
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer
    ///
    /// # Arguments
    /// * `kernel_height` - Height of the pooling kernel
    /// * `kernel_width` - Width of the pooling kernel
    /// * `stride_height` - Stride in height dimension (if None, uses kernel_height)
    /// * `stride_width` - Stride in width dimension (if None, uses kernel_width)
    /// * `padding_height` - Padding in height dimension
    /// * `padding_width` - Padding in width dimension
    /// * `dilation_height` - Dilation in height dimension
    /// * `dilation_width` - Dilation in width dimension
    /// * `return_indices` - Whether to return indices of max values
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_height: usize,
        kernel_width: usize,
        stride_height: Option<usize>,
        stride_width: Option<usize>,
        padding_height: usize,
        padding_width: usize,
        dilation_height: usize,
        dilation_width: usize,
        return_indices: bool,
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
            dilation_height,
            dilation_width,
            return_indices,
        }
    }

    /// Calculate output size for 2D pooling
    fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = ((input_height + 2 * self.padding_height
            - self.dilation_height * (self.kernel_height - 1)
            - 1)
            / self.stride_height)
            + 1;
        let out_width = ((input_width + 2 * self.padding_width
            - self.dilation_width * (self.kernel_width - 1)
            - 1)
            / self.stride_width)
            + 1;

        (out_height, out_width)
    }

    /// Forward pass for 2D max pooling
    fn max_pool2d_forward<T: FloatDtype>(
        &self,
        input: &Tensor<T, CpuBackend>,
    ) -> Result<(Tensor<T, CpuBackend>, Option<Tensor<i32, CpuBackend>>)> {
        let batch_size = input.shape()[0];
        let input_height = input.shape()[1];
        let input_width = input.shape()[2];
        let channels = input.shape()[3];

        let (output_height, output_width) = self.output_size(input_height, input_width);
        let output_shape = vec![batch_size, output_height, output_width, channels];

        let mut output_data = vec![T::zero(); output_shape.iter().product()];
        let mut indices_data = if self.return_indices {
            Some(vec![0i32; output_shape.iter().product()])
        } else {
            None
        };

        // Perform max pooling
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for c in 0..channels {
                        let mut max_val = T::neg_infinity();
                        let mut max_idx = 0i32;

                        // Find max value in kernel window
                        for kh in 0..self.kernel_height {
                            for kw in 0..self.kernel_width {
                                let ih = oh * self.stride_height + kh * self.dilation_height;
                                let iw = ow * self.stride_width + kw * self.dilation_width;

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
                                        let input_val = input.data()[input_idx];

                                        if input_val > max_val {
                                            max_val = input_val;
                                            max_idx = (kh * self.kernel_width + kw) as i32;
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx =
                            ((b * output_height + oh) * output_width + ow) * channels + c;
                        output_data[output_idx] = max_val;

                        if let Some(ref mut indices) = indices_data {
                            indices[output_idx] = max_idx;
                        }
                    }
                }
            }
        }

        let output = Tensor::from_vec(CpuBackend::default(), output_data, output_shape.clone()).unwrap();
        let indices = indices_data.map(|data| Tensor::from_vec(CpuBackend::default(), data, output_shape).unwrap());

        Ok((output, indices))
    }
}

impl<T: FloatDtype> Module<T> for MaxPool2d {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        self.max_pool2d_forward(input)
            .map(|(output, _)| output)
            .map_err(|e| NNError::InvalidInput {
                message: format!("MaxPool2d forward pass failed: {:?}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![] // MaxPool2d has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![] // MaxPool2d has no learnable parameters
    }
}



