use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use storage::{Storage, StorageFromVec, StorageToDense};

use tensor::{ops::TensorStorageOps, tensor_backend_dispatch::TensorBackendDispatcher, Tensor};


/// 1D Transposed Convolutional layer (Deconvolution).
#[derive(Debug, Clone)]
pub struct ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Transposed convolution weights [in_channels, out_channels, kernel_size]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Output padding
    pub output_padding: usize,
    /// Dilation
    pub dilation: usize,
    /// Groups
    pub groups: usize,
    pub(crate) _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        output_padding: Option<usize>,
        groups: Option<usize>,
        dilation: Option<usize>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let output_padding = output_padding.unwrap_or(0);
        let groups = groups.unwrap_or(1);
        let dilation = dilation.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        if in_channels % groups != 0 {
            return Err(crate::core::error::NNError::InvalidInput {
                message: format!(
                    "In channels {} must be divisible by groups {}",
                    in_channels, groups
                ),
            });
        }
        if out_channels % groups != 0 {
            return Err(crate::core::error::NNError::InvalidInput {
                message: format!(
                    "Out channels {} must be divisible by groups {}",
                    out_channels, groups
                ),
            });
        }

        // Initialize weights with Xavier uniform initialization
        // Weight shape for Transposed Conv: [in_channels, out_channels / groups, kernel_size]
        let out_channels_per_group = out_channels / groups;
        let weight_data = Self::xavier_uniform_init(in_channels, out_channels_per_group, kernel_size);
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
            dilation,
            groups,
            _phantom: PhantomData,
        })
    }

    fn xavier_uniform_init(
        in_channels: usize,
        out_channels_per_group: usize,
        kernel_size: usize,
    ) -> Tensor<B, S, T>
    where
        T: num_traits::Float + num_traits::FromPrimitive,
    {
        use rand::distributions::{Distribution, Uniform};
        let shape = [in_channels, out_channels_per_group, kernel_size];
        let total_elements = shape.iter().product();
        let fan_in = total_elements / out_channels_per_group;
        let bound = (6.0 / (fan_in + out_channels_per_group) as f64).sqrt();
        let dist = Uniform::new(-bound, bound);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..total_elements)
            .map(|_| T::from(dist.sample(&mut rng)).unwrap())
            .collect();
        Tensor::<B, S, T>::from_vec(data, &shape).unwrap()
    }

    pub fn output_size(&self, input_length: usize) -> usize {
        (input_length - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) * self.dilation + 1 + self.output_padding
    }

    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Parameter<B, S, T>> {
        self.bias.as_ref()
    }
}

impl<B, S, T> Module<B, S, T> for ConvTranspose1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(crate::core::error::NNError::InvalidInput {
                message: format!("Expected 3D input (N, C, L), got {}D", input_shape.len()),
            });
        }

        let in_channels = input_shape[1];
        if in_channels != self.in_channels {
            return Err(crate::core::error::NNError::InvalidInput {
                message: format!(
                    "Expected {} input channels, got {}",
                    self.in_channels, in_channels
                ),
            });
        }

        let output = tensor::ops::conv::conv_transpose1d(
            input,
            self.weight.data(),
            self.bias.as_ref().map(|b| b.data()),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )?;

        Ok(output)
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

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "ConvTranspose1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
