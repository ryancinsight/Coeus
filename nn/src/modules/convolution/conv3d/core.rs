use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::kernels::conv3d_cpu_dense;

/// 3D Convolutional layer for volumetric feature extraction.
#[derive(Debug, Clone)]
pub struct Conv3D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Convolution weights [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Kernel depth
    pub kernel_depth: usize,
    /// Kernel height
    pub kernel_height: usize,
    /// Kernel width
    pub kernel_width: usize,
    /// Stride in depth dimension
    pub stride_d: usize,
    /// Stride in height dimension
    pub stride_h: usize,
    /// Stride in width dimension
    pub stride_w: usize,
    /// Padding in depth dimension
    pub padding_d: usize,
    /// Padding in height dimension
    pub padding_h: usize,
    /// Padding in width dimension
    pub padding_w: usize,
    pub(crate) _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv3D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
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

    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Parameter<B, S, T>> {
        self.bias.as_ref()
    }

    fn xavier_uniform_init(
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

        let output_cpu = conv3d_cpu_dense(
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
        "Conv3D"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
