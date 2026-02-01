use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use std::ops::Neg;
use storage::{Storage, StorageFromVec, StorageToDense};

use tensor::{ops::TensorStorageOps, tensor_backend_dispatch::TensorBackendDispatcher, Tensor};


/// 1D Convolutional layer.
#[derive(Debug, Clone)]
pub struct Conv1D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Convolution weights [out_channels, in_channels, kernel_size]
    weight: Parameter<B, S, T>,
    /// Bias terms [out_channels] (optional)
    bias: Option<Parameter<B, S, T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    pub(crate) _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv1D<B, S, T>
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

    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Parameter<B, S, T>> {
        self.bias.as_ref()
    }

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
}

impl<B, S, T> Module<B, S, T> for Conv1D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + TensorStorageOps<T> + 'static,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + 'static,
    T: Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

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

        let output = tensor::ops::conv::conv1d(
            input,
            self.weight.data(),
            self.bias.as_ref().map(|b| b.data()),
            self.stride,
            self.padding,
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
        "Conv1D"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
