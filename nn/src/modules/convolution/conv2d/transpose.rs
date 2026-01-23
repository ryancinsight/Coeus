use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use std::ops::Neg;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::kernels::conv_transpose_2d_cpu_dense;

/// 2D Transposed Convolutional layer (Deconvolution).
#[derive(Debug, Clone)]
pub struct ConvTranspose2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Transposed convolution weights [in_channels, out_channels, kernel_height, kernel_width]
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
    /// Output padding in height dimension
    pub output_padding_h: usize,
    /// Output padding in width dimension
    pub output_padding_w: usize,
    pub(crate) _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> ConvTranspose2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        output_padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let (kernel_height, kernel_width) = kernel_size;
        let (stride_h, stride_w) = stride.unwrap_or((1, 1));
        let (padding_h, padding_w) = padding.unwrap_or((0, 0));
        let (output_padding_h, output_padding_w) = output_padding.unwrap_or((0, 0));
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        let weight_data =
            Self::xavier_uniform_init(in_channels, out_channels, kernel_height, kernel_width);
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
            out_channels,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            _phantom: PhantomData,
        })
    }

    fn xavier_uniform_init(
        in_channels: usize,
        out_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
    ) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};
        let shape = [in_channels, out_channels, kernel_height, kernel_width];
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

    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = (input_height - 1) * self.stride_h - 2 * self.padding_h
            + self.kernel_height
            + self.output_padding_h;
        let out_width = (input_width - 1) * self.stride_w - 2 * self.padding_w
            + self.kernel_width
            + self.output_padding_w;
        (out_height, out_width)
    }

    #[must_use]
    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.weight
    }

    #[must_use]
    pub fn bias(&self) -> Option<&Parameter<B, S, T>> {
        self.bias.as_ref()
    }
}

impl<B, S, T> Module<B, S, T> for ConvTranspose2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType
        + FloatExt
        + Neg<Output = T>
        + PartialOrd
        + num_traits::Float
        + num_traits::FromPrimitive
        + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Expected 4D input (N, C, H, W), got {}D", input_shape.len()),
            });
        }

        let in_channels = input_shape[1];
        if in_channels != self.in_channels {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected {} input channels, got {}",
                    self.in_channels, in_channels
                ),
            });
        }

        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;
        let bias_cpu = self
            .bias
            .as_ref()
            .map(|b| b.data().to_cpu_dense())
            .transpose()?;

        let output_cpu = conv_transpose_2d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.output_padding_h,
            self.output_padding_w,
        )?;

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

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "ConvTranspose2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
