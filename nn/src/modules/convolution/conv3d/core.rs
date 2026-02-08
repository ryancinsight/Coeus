use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use storage::{Storage, StorageFromVec, StorageToDense};

use tensor::{ops::TensorStorageOps, tensor_backend_dispatch::TensorBackendDispatcher, Tensor};


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
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType
        + FloatExt
        + PartialOrd
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::Zero
        + 'static,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();
        if input_shape.len() != 5 {
             return Err(NNError::ShapeMismatch {
                operation: "Conv3D forward".to_string(),
                expected: vec![0, 0, 0, 0, 0],
                actual: input_shape.to_vec(),
            });
        }

        let output = crate::functional::convolution::conv3d(
            input,
            self.weight.data(),
            self.bias.as_ref().map(|b| b.data()),
            (self.stride_d, self.stride_h, self.stride_w),
            (self.padding_d, self.padding_h, self.padding_w),
            Some((1, 1, 1)), // dilation
            1, // groups
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

    fn modules(&self) -> Vec<&dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
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

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
