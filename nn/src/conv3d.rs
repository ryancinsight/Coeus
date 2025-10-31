//! 3D Convolutional neural network layers.
//!
//! This module provides 3D convolution operations for building CNNs and processing volumetric data.

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;
use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;
use std::marker::PhantomData;

/// 3D Convolutional layer for volumetric feature extraction.
/// Performs 3D convolution on input tensors of shape [batch_size, in_channels, depth, height, width].
/// Outputs tensors of shape [batch_size, out_channels, out_depth, out_height, out_width].
///
/// # Examples
/// ```rust
/// use nn::{Conv3D, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, 64, (3, 3, 3), None, None, None).unwrap();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[8, 1, 16, 16, 16]).unwrap();
/// let output = conv3d.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[8, 64, 14, 14, 14]);
/// ```
#[derive(Debug, Clone)]
pub struct Conv3D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Convolution weights [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    in_channels: usize,
    /// Kernel depth
    kernel_depth: usize,
    /// Kernel height
    kernel_height: usize,
    /// Kernel width
    kernel_width: usize,
    /// Stride in depth dimension
    stride_d: usize,
    /// Stride in height dimension
    stride_h: usize,
    /// Stride in width dimension
    stride_w: usize,
    /// Padding in depth dimension
    padding_d: usize,
    /// Padding in height dimension
    padding_h: usize,
    /// Padding in width dimension
    padding_w: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv3D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    /// Create a new Conv3D layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - (depth, height, width) of the convolution kernel
    /// * `stride` - (depth, height, width) stride of the convolution (default: (1, 1, 1))
    /// * `padding` - (depth, height, width) padding added to input (default: (0, 0, 0))
    /// * `bias` - Whether to include bias terms (default: true)
    ///
    /// # Examples
    /// ```rust
    /// use nn::Conv3D;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 64, (3, 3, 3), None, None, None).unwrap();
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<(usize, usize, usize)>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let (kernel_depth, kernel_height, kernel_width) = kernel_size;
        let (stride_d, stride_h, stride_w) = stride.unwrap_or((1, 1, 1));
        let (padding_d, padding_h, padding_w) = padding.unwrap_or((0, 0, 0));
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        let weight_data = Self::xavier_uniform_init(
            out_channels * in_channels * kernel_depth * kernel_height * kernel_width,
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        );
        let weight = Parameter::new(weight_data.requires_grad_(true), "weight".to_string());

        let bias_param = if use_bias {
            let zeros_data = vec![T::zero(); out_channels];
            let bias_data = Tensor::<B, S, T>::from_vec(zeros_data, &[out_channels])?;
            Some(Parameter::new(
                bias_data.requires_grad_(true),
                "bias".to_string(),
            ))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            _phantom: PhantomData,
        })
    }

    fn xavier_uniform_init(
        _num_elements: usize,
        out_channels: usize,
        in_channels: usize,
        kernel_depth: usize,
        kernel_height: usize,
        kernel_width: usize,
    ) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};
        let shape = [
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ];
        let total_elements = shape.iter().product();
        let fan_in = total_elements / out_channels;
        let bound = (6.0 / (fan_in + out_channels) as f64).sqrt();
        let dist = Uniform::new(-bound, bound);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..total_elements)
            .map(|_| T::from(dist.sample(&mut rng)).unwrap())
            .collect();
        Tensor::<B, S, T>::from_vec(data, &shape).unwrap()
    }

    /// Compute output size for given input size.
    pub fn output_size(
        &self,
        input_depth: usize,
        input_height: usize,
        input_width: usize,
    ) -> (usize, usize, usize) {
        let out_depth = (input_depth + 2 * self.padding_d - self.kernel_depth) / self.stride_d + 1;
        let out_height =
            (input_height + 2 * self.padding_h - self.kernel_height) / self.stride_h + 1;
        let out_width = (input_width + 2 * self.padding_w - self.kernel_width) / self.stride_w + 1;
        (out_depth, out_height, out_width)
    }

    /// Perform 3D convolution on CPU dense tensors.
    fn conv3d_cpu_dense(
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        padding_d: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let out_channels = weight_shape[0];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        // Calculate output dimensions
        let output_depth = (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1;
        let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

        // Pad input if necessary
        let padded_input = if padding_d > 0 || padding_h > 0 || padding_w > 0 {
            let padded_depth = input_depth + 2 * padding_d;
            let padded_height = input_height + 2 * padding_h;
            let padded_width = input_width + 2 * padding_w;
            let mut padded_data =
                vec![
                    T::zero();
                    batch_size * in_channels * padded_depth * padded_height * padded_width
                ];

            for b in 0..batch_size {
                for c in 0..in_channels {
                    for d in 0..input_depth {
                        for h in 0..input_height {
                            for w in 0..input_width {
                                let input_idx =
                                    (((b * in_channels + c) * input_depth + d) * input_height + h)
                                        * input_width
                                        + w;
                                let padded_idx = (((b * in_channels + c) * padded_depth
                                    + (d + padding_d))
                                    * padded_height
                                    + (h + padding_h))
                                    * padded_width
                                    + (w + padding_w);
                                padded_data[padded_idx] = input.as_slice()[input_idx];
                            }
                        }
                    }
                }
            }
            Tensor::from_vec(
                padded_data,
                &[
                    batch_size,
                    in_channels,
                    padded_depth,
                    padded_height,
                    padded_width,
                ],
            )?
        } else {
            input.clone()
        };

        let padded_shape = padded_input.shape().dims();
        let padded_depth = padded_shape[2];
        let padded_height = padded_shape[3];
        let padded_width = padded_shape[4];

        // Initialize output tensor
        let output_size = batch_size * out_channels * output_depth * output_height * output_width;
        let mut output_data = vec![T::zero(); output_size];

        let input_data = padded_input.as_slice();
        let weight_data = weight.as_slice();

        // Perform 3D convolution
        #[allow(clippy::needless_range_loop)]
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let mut sum = T::zero();

                            // Convolve over input channels and kernel
                            for ic in 0..in_channels {
                                for kd in 0..kernel_depth {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            let id = od * stride_d + kd;
                                            let ih = oh * stride_h + kh;
                                            let iw = ow * stride_w + kw;

                                            if id < padded_depth
                                                && ih < padded_height
                                                && iw < padded_width
                                            {
                                                let input_idx =
                                                    (((b * in_channels + ic) * padded_depth + id)
                                                        * padded_height
                                                        + ih)
                                                        * padded_width
                                                        + iw;
                                                let weight_idx =
                                                    (((oc * in_channels + ic) * kernel_depth + kd)
                                                        * kernel_height
                                                        + kh)
                                                        * kernel_width
                                                        + kw;
                                                sum = sum
                                                    + input_data[input_idx]
                                                        * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            // Add bias if present
                            if let Some(bias_tensor) = bias {
                                let bias_data = bias_tensor.as_slice();
                                sum = sum + bias_data[oc];
                            }

                            let output_idx = (((b * out_channels + oc) * output_depth + od)
                                * output_height
                                + oh)
                                * output_width
                                + ow;
                            output_data[output_idx] = sum;
                        }
                    }
                }
            }
        }

        let output_shape = [
            batch_size,
            out_channels,
            output_depth,
            output_height,
            output_width,
        ];
        Ok(Tensor::from_vec(output_data, &output_shape)?)
    }
}

impl<B, S, T> Module<B, S, T> for Conv3D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType
        + FloatExt
        + PartialOrd
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::Zero
        + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 5usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 5D input (N, C, D, H, W), got {}D",
                    input_shape.len()
                ),
            });
        }

        let _batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let _input_depth = input_shape[2];
        let _input_height = input_shape[3];
        let _input_width = input_shape[4];

        if in_channels != self.in_channels {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected {} input channels, got {}",
                    self.in_channels, in_channels
                ),
            });
        }

        // Convert to CPU dense for computation (maintains compatibility while allowing future generic implementation)
        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;
        let bias_cpu = self
            .bias
            .as_ref()
            .map(|b| b.data().to_cpu_dense())
            .transpose()?;

        // Perform convolution
        let output_cpu = Self::conv3d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.padding_d,
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
        // No-op: Conv3D doesn't have training-specific behavior
    }

    fn name(&self) -> &str {
        "Conv3D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_conv3d_creation() {
        let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            (3, 3, 3),
            Some((1, 1, 1)),
            Some((1, 1, 1)),
            Some(true),
        )
        .unwrap();
        assert_eq!(conv3d.in_channels, 3);
        let weight_shape = conv3d.weight.data().shape().dims();
        assert_eq!(weight_shape[0], 64); // out_channels
        assert_eq!(conv3d.kernel_depth, 3);
        assert_eq!(conv3d.kernel_height, 3);
        assert_eq!(conv3d.kernel_width, 3);
        assert_eq!(conv3d.stride_d, 1);
        assert_eq!(conv3d.stride_h, 1);
        assert_eq!(conv3d.stride_w, 1);
        assert_eq!(conv3d.padding_d, 1);
        assert_eq!(conv3d.padding_h, 1);
        assert_eq!(conv3d.padding_w, 1);
        assert!(conv3d.bias.is_some());
        let params = conv3d.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_conv3d_forward() {
        let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            1,
            2,
            (3, 3, 3),
            Some((1, 1, 1)),
            Some((1, 1, 1)),
            Some(false),
        )
        .unwrap();
        let input_data = vec![Float32::new(1.0); 1 * 1 * 5 * 5 * 5];
        let input = TestTensor::from_vec(input_data, &[1, 1, 5, 5, 5]).unwrap();
        let output = conv3d.forward(&input).unwrap();
        let output_shape = output.shape().dims();
        // With stride=1, padding=1, kernel=3: output = (5 + 2*1 - 3) / 1 + 1 = 5
        assert_eq!(output_shape, &[1, 2, 5, 5, 5]);
    }

    #[test]
    fn test_conv3d_output_size() {
        let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            (3, 3, 3),
            Some((1, 1, 1)),
            Some((1, 1, 1)),
            Some(true),
        )
        .unwrap();
        // Input: 8x8x8 with stride=1, padding=1, kernel=3
        // Output: (8 + 2*1 - 3) / 1 + 1 = 8
        assert_eq!(conv3d.output_size(8, 8, 8), (8, 8, 8));

        let conv3d2 = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            (3, 3, 3),
            Some((2, 2, 2)),
            Some((0, 0, 0)),
            Some(true),
        )
        .unwrap();
        // Input: 16x16x16 with stride=2, padding=0, kernel=3
        // Output: (16 + 2*0 - 3) / 2 + 1 = 7
        assert_eq!(conv3d2.output_size(16, 16, 16), (7, 7, 7));
    }
}
