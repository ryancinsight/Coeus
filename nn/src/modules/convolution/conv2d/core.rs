use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use std::ops::Neg;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::kernels::conv2d_cpu_dense;

/// 2D Convolutional layer for spatial feature extraction.
#[derive(Debug, Clone)]
pub struct Conv2D<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
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
    pub(crate) _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Conv2D<B, S, T>
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
        bias: Option<bool>,
    ) -> Result<Self> {
        let (kernel_height, kernel_width) = kernel_size;
        let (stride_h, stride_w) = stride.unwrap_or((1, 1));
        let (padding_h, padding_w) = padding.unwrap_or((0, 0));
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with Xavier uniform initialization
        let weight_data =
            Self::xavier_uniform_init(out_channels, in_channels, kernel_height, kernel_width);
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
            _phantom: PhantomData,
        })
    }

    fn xavier_uniform_init(
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
        let data: Vec<T> = (0..total_elements)
            .map(|_| T::from(dist.sample(&mut rng)).unwrap())
            .collect();
        Tensor::<B, S, T>::from_vec(data, &shape).unwrap()
    }

    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height =
            (input_height + 2 * self.padding_h - self.kernel_height) / self.stride_h + 1;
        let out_width = (input_width + 2 * self.padding_w - self.kernel_width) / self.stride_w + 1;
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

impl<B, S, T> Module<B, S, T> for Conv2D<B, S, T>
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

        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;
        let bias_cpu = self
            .bias
            .as_ref()
            .map(|b| b.data().to_cpu_dense())
            .transpose()?;

        let output_cpu = conv2d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            self.stride_h,
            self.stride_w,
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
        "Conv2D"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> Conv2D<B, S, T>
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
    #[allow(clippy::type_complexity)]
    pub fn backward(
        &self,
        grad_output: &Tensor<B, S, T>,
        input: &Tensor<B, S, T>,
    ) -> Result<(
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        Option<Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    )> {
        let grad_output_cpu = grad_output.to_cpu_dense()?;
        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = self.weight.data().to_cpu_dense()?;

        let input_grad_cpu = crate::functional_api::conv_transpose_2d(
            &grad_output_cpu,
            &weight_cpu,
            None,
            Some((self.stride_h, self.stride_w)),
            Some((self.padding_h, self.padding_w)),
            Some((0, 0)),
        )?;

        let weight_grad_cpu = self.compute_weight_gradients(&grad_output_cpu, &input_cpu)?;

        let bias_grad_cpu = if self.bias.is_some() {
            Some(self.compute_bias_gradients(&grad_output_cpu)?)
        } else {
            None
        };

        Ok((input_grad_cpu, weight_grad_cpu, bias_grad_cpu))
    }

    fn compute_weight_gradients(
        &self,
        grad_output: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
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
                                    let ih = oh as isize * self.stride_h as isize + kh as isize
                                        - self.padding_h as isize;
                                    let iw = ow as isize * self.stride_w as isize + kw as isize
                                        - self.padding_w as isize;
                                    if ih >= 0
                                        && ih < in_height as isize
                                        && iw >= 0
                                        && iw < in_width as isize
                                    {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let input_idx = ((b * in_channels + ic) * in_height + ih)
                                            * in_width
                                            + iw;
                                        let grad_idx = ((b * out_channels + oc) * out_height + oh)
                                            * out_width
                                            + ow;
                                        sum = sum
                                            + input_data[input_idx] * grad_output_data[grad_idx];
                                    }
                                }
                            }
                        }
                        let weight_idx = ((oc * in_channels + ic) * self.kernel_height + kh)
                            * self.kernel_width
                            + kw;
                        weight_grad_data[weight_idx] = sum;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(
            weight_grad_data,
            &[
                out_channels,
                in_channels,
                self.kernel_height,
                self.kernel_width,
            ],
        )?)
    }

    fn compute_bias_gradients(
        &self,
        grad_output: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
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
