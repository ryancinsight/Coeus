//! 1D Convolutional neural network layers.
//!
//! This module provides 1D convolution operations for building CNNs and processing sequential data.

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;
use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;
use std::marker::PhantomData;

/// 1D Convolutional layer.
///
/// Performs 1D convolution on input tensors of shape [batch_size, in_channels, length].
/// Outputs tensors of shape [batch_size, out_channels, out_length].
///
/// This is essential for audio processing, speech recognition, time-series analysis,
/// and 1D signal processing.
///
/// # Shape
/// - Input: `(N, C_in, L_in)` where N is batch size, C_in is input channels, L_in is input length
/// - Output: `(N, C_out, L_out)` where C_out is output channels, L_out is output length
///
/// # Examples
/// ```rust
/// use nn::{Conv1D, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Audio processing: 1 channel input, 64 filters, kernel size 3
/// let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, 64, 3, None, None, None).unwrap();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 1000]).unwrap();
/// let output = conv.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 998]);
/// ```
///
/// # References
/// - van den Oord et al. (2016): "WaveNet: A Generative Model for Raw Audio"
/// - Collobert & Weston (2008): "A unified architecture for natural language processing"
#[derive(Debug, Clone)]
pub struct Conv1D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Convolution weights [out_channels, in_channels, kernel_size]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    in_channels: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv1D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    /// Create a new Conv1D layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding added to input (default: 0)
    /// * `bias` - Whether to include bias terms (default: true)
    ///
    /// # Examples
    /// ```rust
    /// use nn::Conv1D;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, 64, 3, None, None, None).unwrap();
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        let weight_data =
            Self::xavier_uniform_init(&[out_channels, in_channels, kernel_size], out_channels);
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
            stride,
            padding,
            _phantom: PhantomData,
        })
    }

    /// Xavier uniform initialization for weights.
    fn xavier_uniform_init(shape: &[usize], fan_out: usize) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};

        let mut rng = rand::thread_rng();
        let num_elements = shape.iter().product();
        let fan_in = num_elements / fan_out;
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let dist = Uniform::new(-bound, bound);

        let data: Vec<T> = (0..num_elements)
            .map(|_| T::from(dist.sample(&mut rng)).unwrap())
            .collect();

        Tensor::from_vec(data, shape).unwrap()
    }

    /// Compute output length after convolution.
    fn compute_output_length(
        input_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> usize {
        (input_length + 2 * padding - kernel_size) / stride + 1
    }

    /// Perform 1D convolution on CPU dense tensors.
    fn conv1d_cpu_dense(
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_length = input_shape[2];
        let out_channels = weight_shape[0];
        let kernel_size = weight_shape[2];

        // Validate dimensions
        if weight_shape[1] != in_channels {
            return Err(NNError::ShapeMismatch {
                operation: "Conv1D".to_string(),
                expected: vec![out_channels, in_channels, kernel_size],
                actual: weight_shape.to_vec(),
            });
        }

        let output_length = Self::compute_output_length(input_length, kernel_size, stride, padding);

        // Initialize output tensor
        let output_size = batch_size * out_channels * output_length;
        let mut output_data = vec![T::zero(); output_size];

        // Pad input if necessary
        let padded_length = input_length + 2 * padding;
        let mut padded_input = vec![T::zero(); batch_size * in_channels * padded_length];

        if padding > 0 {
            // Copy input to padded tensor with padding
            for b in 0..batch_size {
                for c in 0..in_channels {
                    for l in 0..input_length {
                        let input_idx = ((b * in_channels + c) * input_length) + l;
                        let padded_idx = ((b * in_channels + c) * padded_length) + l + padding;
                        padded_input[padded_idx] = input.as_slice()[input_idx];
                    }
                }
            }
        } else {
            // No padding, just copy
            padded_input.copy_from_slice(input.as_slice());
        }

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for ol in 0..output_length {
                    let mut sum = T::zero();

                    for ic in 0..in_channels {
                        for k in 0..kernel_size {
                            let input_pos = ol * stride + k;
                            if input_pos < padded_length {
                                let input_idx =
                                    ((b * in_channels + ic) * padded_length) + input_pos;
                                let weight_idx = ((oc * in_channels + ic) * kernel_size) + k;
                                sum = sum + padded_input[input_idx] * weight.as_slice()[weight_idx];
                            }
                        }
                    }

                    // Add bias if provided
                    if let Some(bias_tensor) = bias {
                        sum = sum + bias_tensor.as_slice()[oc];
                    }

                    let output_idx = ((b * out_channels + oc) * output_length) + ol;
                    output_data[output_idx] = sum;
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            &[batch_size, out_channels, output_length],
        )?)
    }
}

impl<B, S, T> Module<B, S, T> for Conv1D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        // Validate input shape: [batch_size, in_channels, length]
        if input_shape.len() != 3usize {
            return Err(NNError::ShapeMismatch {
                operation: "Conv1D forward".to_string(),
                expected: vec![0, self.in_channels, 0],
                actual: input_shape.to_vec(),
            });
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_length = input_shape[2];

        if in_channels != self.in_channels {
            return Err(NNError::ShapeMismatch {
                operation: "Conv1D forward".to_string(),
                expected: vec![batch_size, self.in_channels, input_length],
                actual: input_shape.to_vec(),
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
        let output_cpu = Self::conv1d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride,
            self.padding,
        )?;

        // Convert back to original backend/storage
        let output_shape = output_cpu.shape().dims();
        let output_data = output_cpu.as_slice().to_vec();

        // Create output tensor in original backend
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
        // No-op: Conv1D doesn't have training-specific behavior
    }

    fn name(&self) -> &str {
        "Conv1D"
    }
}

/// 1D Transposed Convolutional layer (Deconvolution).
///
/// Performs 1D transposed convolution (also known as deconvolution or fractionally-strided convolution)
/// on input tensors of shape [batch_size, in_channels, length].
/// Outputs tensors of shape [batch_size, out_channels, out_length].
///
/// Transposed convolution is the mathematical inverse of convolution and is used for upsampling
/// in audio generation, 1D signal super-resolution, and temporal upsampling tasks.
///
/// # Output Size Formula
/// ```text
/// out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding
/// ```
///
/// # Shape
/// - Input: `(N, C_in, L_in)` where N is batch size, C_in is input channels, L_in is input length
/// - Output: `(N, C_out, L_out)` where C_out is output channels, L_out is output length
///
/// # Examples
/// ```rust
/// use nn::{ConvTranspose1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Audio upsampling: 64 channels input, 1 channel output, kernel size 4, stride 2
/// let conv_transpose = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 1, 4, Some(2), Some(1), Some(0), Some(true)).unwrap();
/// // Layer created successfully
/// ```
///
/// # References
/// - Dumoulin & Visin (2016): "A guide to convolution arithmetic for deep learning"
/// - van den Oord et al. (2016): "WaveNet: A Generative Model for Raw Audio"
#[derive(Debug, Clone)]
pub struct ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Transposed convolution weights [in_channels, out_channels, kernel_size]
    /// Note: Weight shape is reversed compared to Conv1D
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    in_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Output padding
    output_padding: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    /// Create a new ConvTranspose1d layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding added to input (default: 0)
    /// * `output_padding` - Additional size added to output (default: 0)
    /// * `bias` - Whether to include bias terms (default: true)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        output_padding: Option<usize>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let output_padding = output_padding.unwrap_or(0);
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        // Weight shape: [in_channels, out_channels, kernel_size]
        let weight_data = Self::xavier_uniform_init(in_channels, out_channels, kernel_size);
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
            kernel_size,
            stride,
            padding,
            output_padding,
            _phantom: PhantomData,
        })
    }

    /// Xavier uniform initialization for weights.
    fn xavier_uniform_init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};

        let shape = [in_channels, out_channels, kernel_size];
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
    pub fn output_size(&self, input_length: usize) -> usize {
        (input_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
    }

    /// Perform 1D transposed convolution on CPU dense tensors.
    fn conv_transpose_1d_cpu_dense(
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        bias: Option<&Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_length = input_shape[2];
        let out_channels = weight_shape[1];
        let kernel_size = weight_shape[2];

        let output_length =
            (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

        // Initialize output tensor
        let output_size = batch_size * out_channels * output_length;
        let mut output_data = vec![T::zero(); output_size];

        // Perform transposed convolution
        for b in 0..batch_size {
            for ic in 0..in_channels {
                for il in 0..input_length {
                    for oc in 0..out_channels {
                        for k in 0..kernel_size {
                            // Calculate output position: careful with underflow
                            let stride_term = il * stride;
                            let kernel_term = k;
                            let padding_term = padding;

                            // Check bounds to prevent underflow
                            if stride_term + kernel_term >= padding_term {
                                let output_pos = stride_term + kernel_term - padding_term;
                                if output_pos < output_length {
                                    let input_idx = ((b * in_channels + ic) * input_length) + il;
                                    let weight_idx = ((ic * out_channels + oc) * kernel_size) + k;
                                    let output_idx =
                                        ((b * out_channels + oc) * output_length) + output_pos;

                                    output_data[output_idx] = output_data[output_idx]
                                        + input.as_slice()[input_idx]
                                            * weight.as_slice()[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            for b in 0..batch_size {
                for oc in 0..out_channels {
                    for ol in 0..output_length {
                        let output_idx = ((b * out_channels + oc) * output_length) + ol;
                        output_data[output_idx] =
                            output_data[output_idx] + bias_tensor.as_slice()[oc];
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            &[batch_size, out_channels, output_length],
        )?)
    }
}

impl<B, S, T> Module<B, S, T> for ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: format!("Expected 3D input (N, C, L), got {}D", input_shape.len()),
            });
        }

        let _batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let _input_length = input_shape[2];

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

        // Perform transposed convolution
        let output_cpu = Self::conv_transpose_1d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
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

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // No-op: ConvTranspose1d doesn't have training-specific behavior
    }

    fn name(&self) -> &str {
        "ConvTranspose1d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_conv1d_creation() {
        let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            5,
            Some(1),
            Some(2),
            Some(true),
        )
        .unwrap();

        assert_eq!(conv.in_channels, 3);
        let weight_shape = conv.weight.data().shape().dims();
        assert_eq!(weight_shape[0], 64); // out_channels
        assert_eq!(weight_shape[2], 5); // kernel_size
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 2);
        assert!(conv.bias.is_some());

        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].name(), "weight");
        assert_eq!(params[1].name(), "bias");
    }

    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            1,
            2,
            3,
            Some(1),
            Some(1),
            Some(false),
        )
        .unwrap();

        // Input: [batch_size=1, channels=1, length=5]
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 5],
        )
        .unwrap();

        let output = conv.forward(&input).unwrap();
        let output_shape = output.shape().dims();

        // Expected: [batch_size=1, channels=2, length=5] (with stride=1, padding=1, kernel=3)
        // Output length = (5 + 2*1 - 3) / 1 + 1 = 5
        assert_eq!(output_shape, &[1, 2, 5]);
    }

    #[test]
    fn test_conv_transpose_1d_creation() {
        let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            64,
            1,
            4,
            Some(2),
            Some(1),
            Some(0),
            Some(true),
        )
        .unwrap();

        assert_eq!(conv.in_channels, 64);
        let weight_shape = conv.weight.data().shape().dims();
        assert_eq!(weight_shape[1], 1); // out_channels
        assert_eq!(conv.kernel_size, 4);
        assert_eq!(conv.stride, 2);
        assert_eq!(conv.padding, 1);
        assert_eq!(conv.output_padding, 0);
        assert!(conv.bias.is_some());
    }

    #[test]
    fn test_conv_transpose_1d_forward() {
        let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            2,
            3,
            3,
            Some(2),
            Some(1),
            Some(0),
            Some(false),
        )
        .unwrap();

        // Input: [batch_size=1, channels=2, length=4]
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 2, 4],
        )
        .unwrap();

        let output = conv.forward(&input).unwrap();
        let output_shape = output.shape().dims();

        // Expected output length: (4 - 1) * 2 - 2 * 1 + 3 + 0 = 7
        assert_eq!(output_shape, &[1, 3, 7]);
    }

    #[test]
    fn test_conv1d_output_length_calculation() {
        // Test various configurations
        assert_eq!(
            Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::compute_output_length(
                10, 3, 1, 0
            ),
            8
        );
        assert_eq!(
            Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::compute_output_length(
                10, 3, 1, 1
            ),
            10
        );
        assert_eq!(
            Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::compute_output_length(
                10, 3, 2, 0
            ),
            4
        );
    }

    #[test]
    fn test_conv_transpose_1d_output_size() {
        let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            1,
            1,
            4,
            Some(2),
            Some(1),
            Some(0),
            Some(true),
        )
        .unwrap();
        assert_eq!(conv.output_size(100), 200); // (100 - 1) * 2 - 2 * 1 + 4 + 0 = 200
    }
}
