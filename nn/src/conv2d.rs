//! 2D Convolutional neural network layers.
//!
//! This module provides 2D convolution operations for building CNNs and processing image data.

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;
use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{Storage, DenseStorage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;
use std::marker::PhantomData;
use std::ops::Neg;

/// 2D Convolutional layer for spatial feature extraction.
/// Performs 2D convolution on input tensors of shape [batch_size, in_channels, height, width].
/// Outputs tensors of shape [batch_size, out_channels, out_height, out_width].
///
/// # Examples
/// ```rust
/// use coeus_nn::{Conv2D, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3), None, None, None).unwrap();
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[32, 3, 32, 32]).unwrap();
/// let output = conv.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[32, 64, 30, 30]);
/// ```
#[derive(Debug, Clone)]
pub struct Conv2D<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Convolution weights [out_channels, in_channels, kernel_height, kernel_width]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in height dimension
    pub stride_h: usize,
    /// Stride in width dimension
    pub stride_w: usize,
    /// Padding in height dimension
    pub padding_h: usize,
    /// Padding in width dimension
    pub padding_w: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv2D<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    /// Create a new Conv2D layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - (height, width) of the convolution kernel
    /// * `stride` - (height, width) stride of the convolution (default: (1, 1))
    /// * `padding` - (height, width) padding added to input (default: (0, 0))
    /// * `bias` - Whether to include bias terms (default: true)
    ///
/// # Examples
/// ```rust
/// use coeus_nn::Conv2D;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
    ///
    /// let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3), None, None, None).unwrap();
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let (kernel_height, kernel_width) = kernel_size;
        let (stride_h, stride_w) = stride.unwrap_or((1, 1));
        let (padding_h, padding_w) = padding.unwrap_or((0, 0));
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        let weight_data = Self::xavier_uniform_init(
            out_channels * in_channels * kernel_height * kernel_width,
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        );
        let weight = Parameter::new(weight_data.requires_grad_(true), "weight".to_string());

        let bias_param = if use_bias {
            let zeros_data = vec![T::zero(); out_channels];
            let bias_data = Tensor::<B, S, T>::from_vec(zeros_data, &[out_channels])?;
            Some(Parameter::new(bias_data.requires_grad_(true), "bias".to_string()))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            _phantom: PhantomData,
        })
    }

    fn xavier_uniform_init(
        _num_elements: usize,
        out_channels: usize,
        in_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
    ) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};
        let shape = [out_channels, in_channels, kernel_height, kernel_width];
        let total_elements = shape.iter().product();
        let fan_in = total_elements / out_channels;
        let bound = (6.0 / (fan_in + out_channels) as f64).sqrt();
        let dist = Uniform::new(-bound, bound);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..total_elements).map(|_| T::from(dist.sample(&mut rng)).unwrap()).collect();
        Tensor::<B, S, T>::from_vec(data, &shape).unwrap()
    }

    /// Get the output dimensions for given input dimensions.
    ///
    /// # Arguments
    /// * `input_height` - Input height
    /// * `input_width` - Input width
    ///
    /// # Returns
    /// (output_height, output_width)
    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height =
            (input_height + 2 * self.padding_h - self.kernel_height) / self.stride_h + 1;
        let out_width = (input_width + 2 * self.padding_w - self.kernel_width) / self.stride_w + 1;
        (out_height, out_width)
    }

    /// Pad input tensor with zeros according to padding_h and padding_w
    fn pad_input(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let padded_height = input_height + 2 * self.padding_h;
        let padded_width = input_width + 2 * self.padding_w;

        let input_data = input.as_slice();
        let mut padded_data =
            vec![T::zero(); batch_size * in_channels * padded_height * padded_width];

        // Copy input data to padded tensor with offset
        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..input_height {
                    for w in 0..input_width {
                        let input_idx =
                            ((b * in_channels + c) * input_height + h) * input_width + w;
                        let padded_idx = ((b * in_channels + c) * padded_height
                            + (h + self.padding_h))
                            * padded_width
                            + (w + self.padding_w);
                        padded_data[padded_idx] = input_data[input_idx];
                    }
                }
            }
        }

        let padded_shape = [batch_size, in_channels, padded_height, padded_width];
        Tensor::from_vec(padded_data, &padded_shape).map_err(Into::into)
    }

    fn conv2d_cpu_dense(
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend, DenseStorage<T>, T>,
        bias: Option<&Tensor<CpuBackend, DenseStorage<T>, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let out_channels = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        // Calculate output dimensions
        let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

        // Pad input if necessary
        let padded_input = if padding_h > 0 || padding_w > 0 {
            let padded_height = input_height + 2 * padding_h;
            let padded_width = input_width + 2 * padding_w;
            let mut padded_data = vec![T::zero(); batch_size * in_channels * padded_height * padded_width];

            for b in 0..batch_size {
                for c in 0..in_channels {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            let input_idx = ((b * in_channels + c) * input_height + h) * input_width + w;
                            let padded_idx = ((b * in_channels + c) * padded_height + (h + padding_h)) * padded_width + (w + padding_w);
                            padded_data[padded_idx] = input.as_slice()[input_idx];
                        }
                    }
                }
            }
            Tensor::from_vec(padded_data, &[batch_size, in_channels, padded_height, padded_width])?
        } else {
            input.clone()
        };

        let padded_shape = padded_input.shape().dims();
        let padded_height = padded_shape[2];
        let padded_width = padded_shape[3];

        // Initialize output tensor
        let output_size = batch_size * out_channels * output_height * output_width;
        let mut output_data = vec![T::zero(); output_size];

        let input_data = padded_input.as_slice();
        let weight_data = weight.as_slice();

        // Perform convolution
        #[allow(clippy::needless_range_loop)]
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = T::zero();

                        // Convolve over input channels, kernel height, kernel width
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    // Input position (accounting for stride)
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;

                                    // Input data index
                                    let input_idx = ((b * in_channels + ic) * padded_height + ih)
                                        * padded_width
                                        + iw;
                                    let input_val = input_data[input_idx];

                                    // Weight data index
                                    let weight_idx = ((oc * in_channels + ic) * kernel_height
                                        + kh)
                                        * kernel_width
                                        + kw;
                                    let weight_val = weight_data[weight_idx];

                                    sum = sum + input_val * weight_val;
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_tensor) = bias {
                            let bias_data = bias_tensor.as_slice();
                            sum = sum + bias_data[oc];
                        }

                        // Output data index
                        let output_idx =
                            ((b * out_channels + oc) * output_height + oh) * output_width + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        let output_shape = [batch_size, out_channels, output_height, output_width];
        Ok(Tensor::from_vec(output_data, &output_shape)?)
    }
}

impl<B, S, T> Module<B, S, T> for Conv2D<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + Neg<Output = T> + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        // Validate input shape: [batch_size, in_channels, height, width]
        if input_shape.len() != 4 {
            return Err(NNError::ShapeMismatch {
                operation: "Conv2D forward".to_string(),
                expected: vec![0, self.in_channels, 0, 0],
                actual: input_shape.to_vec(),
            });
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        if in_channels != self.in_channels {
            return Err(NNError::ShapeMismatch {
                operation: "Conv2D forward".to_string(),
                expected: vec![batch_size, self.in_channels, input_height, input_width],
                actual: input_shape.to_vec(),
            });
        }

        let (output_height, output_width) = self.output_size(input_height, input_width);

        // Convert to CPU dense for computation (maintains compatibility while allowing future generic implementation)
        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;
        let bias_cpu = self.bias.as_ref().map(|b| b.data().to_cpu_dense()).transpose()?;

        // Perform convolution
        let output_cpu = Self::conv2d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
        )?;

        // Convert back to original backend/storage
        let output_shape = output_cpu.shape().dims();
        let output_data = output_cpu.as_slice().to_vec();

        Ok(Tensor::from_vec(output_data, output_shape)?)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // Training mode doesn't affect Conv2D currently
    }

    fn name(&self) -> &str {
        "Conv2D"
    }
}

impl<B, S, T> Conv2D<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + Neg<Output = T> + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    /// Compute gradients for Conv2D backward pass.
    ///
    /// Given the gradient with respect to the output, computes gradients with respect to:
    /// - Input tensor
    /// - Weight tensor
    /// - Bias tensor (if present)
    ///
    /// # Arguments
    /// * `grad_output` - Gradient with respect to convolution output [batch, out_channels, out_height, out_width]
    /// * `input` - Original input tensor [batch, in_channels, in_height, in_width]
    ///
    /// # Returns
    /// Tuple of (input_grad, weight_grad, bias_grad) where bias_grad is None if no bias
    ///
    /// # Errors
    /// Returns error if tensor shapes are incompatible
    pub fn backward(
        &self,
        grad_output: &Tensor<B, S, T>,
        input: &Tensor<B, S, T>,
    ) -> Result<(Tensor<CpuBackend, DenseStorage<T>, T>, Tensor<CpuBackend, DenseStorage<T>, T>, Option<Tensor<CpuBackend, DenseStorage<T>, T>>)> {
        // Convert to CPU dense for computation (generic implementation would require backend-specific kernels)
        let grad_output_cpu = grad_output.to_cpu_dense()?;
        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;

        // Compute input gradients using transposed convolution
        let input_grad_cpu = crate::functional::conv_transpose_2d(
            &grad_output_cpu,
            &weight_cpu,
            Some((self.stride_h, self.stride_w)),
            Some((self.padding_h, self.padding_w)),
            Some((0, 0)), // output_padding not supported yet
        )?;

        // Compute weight gradients using cross-correlation
        let weight_grad_cpu = self.compute_weight_gradients(&grad_output_cpu, &input_cpu)?;

        // Compute bias gradients using sum reduction
        let bias_grad_cpu = if self.bias.is_some() {
            Some(self.compute_bias_gradients(&grad_output_cpu)?)
        } else {
            None
        };

        // Return CPU dense gradients (full backend support requires generic computation)
        Ok((input_grad_cpu, weight_grad_cpu, bias_grad_cpu))
    }

    /// Compute weight gradients using cross-correlation between input and output gradients.
    fn compute_weight_gradients(
        &self,
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let grad_output_shape = grad_output.shape().dims();
        let input_shape = input.shape().dims();
        let batch_size = grad_output_shape[0];
        let out_channels = grad_output_shape[1];
        let out_height = grad_output_shape[2];
        let out_width = grad_output_shape[3];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let grad_output_data = grad_output.as_slice();
        let input_data = input.as_slice();
        let weight_size = out_channels * in_channels * self.kernel_height * self.kernel_width;
        let mut weight_grad_data = vec![T::zero(); weight_size];

        #[allow(clippy::needless_range_loop)]
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for kh in 0..self.kernel_height {
                    for kw in 0..self.kernel_width {
                        let mut sum = T::zero();
                        for b in 0..batch_size {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    let ih = oh as isize * self.stride_h as isize + kh as isize - self.padding_h as isize;
                                    let iw = ow as isize * self.stride_w as isize + kw as isize - self.padding_w as isize;
                                    if ih >= 0 && ih < in_height as isize && iw >= 0 && iw < in_width as isize {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                        let grad_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                                        sum = sum + input_data[input_idx] * grad_output_data[grad_idx];
                                    }
                                }
                            }
                        }
                        let weight_idx = ((oc * in_channels + ic) * self.kernel_height + kh) * self.kernel_width + kw;
                        weight_grad_data[weight_idx] = sum;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(weight_grad_data, &[out_channels, in_channels, self.kernel_height, self.kernel_width])?)
    }

    /// Compute bias gradients using sum reduction over spatial and batch dimensions.
    fn compute_bias_gradients(
        &self,
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let grad_output_shape = grad_output.shape().dims();
        let out_channels = grad_output_shape[1];
        let out_height = grad_output_shape[2];
        let out_width = grad_output_shape[3];
        let grad_output_data = grad_output.as_slice();
        let mut bias_grad_data = vec![T::zero(); out_channels];

        #[allow(clippy::needless_range_loop)]
        for oc in 0..out_channels {
            let mut sum = T::zero();
            for b in 0..grad_output_shape[0] {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        sum = sum + grad_output_data[idx];
                    }
                }
            }
            bias_grad_data[oc] = sum;
        }
        Ok(Tensor::from_vec(bias_grad_data, &[out_channels])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    type TestTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3), Some((1, 1)), Some((1, 1)), Some(true)).unwrap();
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 64);
        assert_eq!(conv.kernel_height, 3);
        assert_eq!(conv.kernel_width, 3);
        assert_eq!(conv.stride_h, 1);
        assert_eq!(conv.stride_w, 1);
        assert_eq!(conv.padding_h, 1);
        assert_eq!(conv.padding_w, 1);
        assert!(conv.bias.is_some());
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(1, 2, (3, 3), Some((1, 1)), Some((1, 1)), Some(false)).unwrap();
        let input_data = vec![Float32::new(1.0); 25];
        let input = TestTensor::from_vec(input_data, &[1, 1, 5, 5]).unwrap();
        let output = conv.forward(&input).unwrap();
        let output_shape = output.shape().dims();
        assert_eq!(output_shape, &[1, 2, 5, 5]);
    }

    #[test]
    fn test_conv2d_output_size() {
        let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3), Some((1, 1)), Some((1, 1)), Some(true)).unwrap();
        assert_eq!(conv.output_size(32, 32), (32, 32));
        let conv2 = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3), Some((2, 2)), Some((0, 0)), Some(true)).unwrap();
        assert_eq!(conv2.output_size(28, 28), (13, 13));
    }

    #[test]
    fn test_conv2d_backward_basic() {
        let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(1, 1, (3, 3), None, None, Some(true)).unwrap();
        let input = TestTensor::from_vec(vec![Float32::new(1.0); 25], &[1, 1, 5, 5]).unwrap();
        let grad_output = TestTensor::from_vec(vec![Float32::new(1.0); 9], &[1, 1, 3, 3]).unwrap();
        let (input_grad, weight_grad, bias_grad) = conv.backward(&grad_output, &input).unwrap();
        assert_eq!(input_grad.shape().dims(), &[1, 1, 5, 5]);
        assert_eq!(weight_grad.shape().dims(), &[1, 1, 3, 3]);
        assert_eq!(bias_grad.as_ref().unwrap().shape().dims(), &[1]);
    }

    #[test]
    fn test_conv2d_backward_no_bias() {
        let conv = Conv2D::<CpuBackend, DenseStorage<Float32>, Float32>::new(2, 3, (2, 2), Some((1, 1)), Some((1, 1)), Some(false)).unwrap();
        let input = TestTensor::from_vec(vec![Float32::new(0.5); 64], &[2, 2, 4, 4]).unwrap();
        let grad_output = TestTensor::from_vec(vec![Float32::new(1.0); 96], &[2, 3, 4, 4]).unwrap();
        let (input_grad, weight_grad, bias_grad) = conv.backward(&grad_output, &input).unwrap();
        assert_eq!(input_grad.shape().dims(), &[2, 2, 4, 4]);
        assert_eq!(weight_grad.shape().dims(), &[3, 2, 2, 2]);
        assert!(bias_grad.is_none());
    }
}
