//! Fake quantization for quantization-aware training

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

use crate::quantization::core::{QuantizationGranularity, QuantizationScheme};
use crate::quantization::quantization_ops::QuantizationOps;

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

/// Fake quantization operation for quantization-aware training
///
/// This module simulates quantization effects during training by applying
/// quantization to inputs during forward pass while maintaining gradient flow.
/// Supports both per-tensor and per-channel quantization for improved accuracy.
///
/// # Generic Parameters
/// - `B`: Backend type
/// - `S`: Storage type for parameters
/// - `T`: Data type for parameters and computation
/// - `BITS`: Quantization bitwidth (4, 8, or 16)
#[derive(Debug)]
pub struct FakeQuantize<B, S, T, const BITS: usize>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Quantization scale parameter(s) - can be per-tensor or per-channel
    pub scale: Parameter<B, S, T>,
    /// Quantization zero point parameter(s) - can be per-tensor or per-channel
    pub zero_point: Parameter<B, S, T>,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Number of channels (1 for per-tensor, >1 for per-channel)
    pub num_channels: usize,
    /// Phantom data for backend and storage types
    _phantom: core::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T, const BITS: usize> FakeQuantize<B, S, T, BITS>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + num_traits::Float + num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    /// Create a new fake quantization module
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `scheme` - Quantization scheme (Affine or Symmetric)
    /// * `granularity` - Quantization granularity (PerTensor or PerChannel)
    /// * `num_channels` - Number of channels (1 for per-tensor, >1 for per-channel)
    ///
    /// # Returns
    /// Fake quantization module
    pub fn new(
        backend: B,
        scheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        num_channels: usize,
    ) -> Result<Self> {
        // Validate bitwidth at compile time
        if BITS != 4 && BITS != 8 && BITS != 16 {
            return Err(NNError::InvalidInput {
                message: format!("Unsupported bitwidth: {}. Supported: 4, 8, 16", BITS),
            });
        }

        // Validate parameters
        if num_channels == 0 {
            return Err(NNError::InvalidInput {
                message: "num_channels must be > 0".to_string(),
            });
        }

        match granularity {
            QuantizationGranularity::PerTensor => {
                if num_channels != 1 {
                    return Err(NNError::InvalidInput {
                        message: "PerTensor granularity requires num_channels = 1".to_string(),
                    });
                }
            }
            QuantizationGranularity::PerChannel => {
                if num_channels < 2 {
                    return Err(NNError::InvalidInput {
                        message: "PerChannel granularity requires num_channels >= 2".to_string(),
                    });
                }
            }
        }

        // Create parameter tensors based on granularity
        let param_shape = match granularity {
            QuantizationGranularity::PerTensor => vec![1],
            QuantizationGranularity::PerChannel => vec![num_channels],
        };

        let scale_tensor = Tensor::<B, S, T>::from_vec(
            vec![T::one(); param_shape.iter().product()],
            &param_shape,
            backend.clone(),
        )?;
        let scale = Parameter::new(scale_tensor.requires_grad_(true), "scale".to_string());

        let zero_point_tensor = Tensor::<B, S, T>::from_vec(
            vec![T::zero(); param_shape.iter().product()],
            &param_shape,
            backend,
        )?;
        let zero_point = Parameter::new(
            zero_point_tensor.requires_grad_(true),
            "zero_point".to_string(),
        );

        Ok(Self {
            scale,
            zero_point,
            scheme,
            granularity,
            num_channels,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Forward pass with fake quantization
    ///
    /// # Arguments
    /// * `input` - Input tensor to quantize
    ///
    /// # Returns
    /// Fake quantized output (still floating-point for gradient flow)
    pub fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let scales = self.scale.data().as_slice();
        let zero_points = self.zero_point.data().as_slice();
        self.fake_quantize(input, scales, zero_points)
    }

    /// Apply fake quantization to a tensor
    ///
    /// Simulates quantization effects by:
    /// 1. Quantizing using the selected scheme and granularity
    /// 2. Clamping to quantization range [qmin, qmax]
    /// 3. Dequantizing (maintains gradient flow via STE)
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `scales` - Quantization scale(s) (per-tensor or per-channel)
    /// * `zero_points` - Quantization zero point(s) (per-tensor or per-channel)
    ///
    /// # Returns
    /// Fake quantized tensor (maintains gradient flow via STE)
    fn fake_quantize(
        &self,
        x: &Tensor<B, S, T>,
        scales: &[T],
        zero_points: &[T],
    ) -> Result<Tensor<B, S, T>>
    where
        T: Clone + PartialOrd,
    {
        // Calculate quantization range based on bitwidth
        let (qmin, qmax) = Self::quantization_range();

        // Get input data
        let x_data = x.as_slice();
        let input_shape = x.shape().dims();
        let mut quantized_data = Vec::with_capacity(x_data.len());

        // Apply fake quantization based on granularity
        match self.granularity {
            QuantizationGranularity::PerTensor => {
                // Single scale/zero_point for entire tensor
                let scale = &scales[0];
                let zero_point = &zero_points[0];

                for val in x_data {
                    let quantized = self.quantize_value(val, scale, zero_point, qmin, qmax)?;
                    let dequantized =
                        self.dequantize_value(&quantized, scale, zero_point, qmin, qmax)?;
                    quantized_data.push(dequantized);
                }
            }
            QuantizationGranularity::PerChannel => {
                // Per-channel quantization - determine channel dimension
                // Assume channels are the last dimension for simplicity
                // (can be extended for different channel layouts)
                let channel_dim = input_shape.len() - 1;
                let channels_per_sample = input_shape[channel_dim];

                if channels_per_sample != self.num_channels {
                    return Err(NNError::InvalidInput {
                        message: format!(
                            "Input has {} channels but quantization expects {}",
                            channels_per_sample, self.num_channels
                        ),
                    });
                }

                // Apply per-channel quantization
                let mut data_idx = 0;
                for _ in 0..x_data.len() / channels_per_sample {
                    for channel in 0..channels_per_sample {
                        let val = &x_data[data_idx];
                        let scale = &scales[channel];
                        let zero_point = &zero_points[channel];

                        let quantized = self.quantize_value(val, scale, zero_point, qmin, qmax)?;
                        let dequantized =
                            self.dequantize_value(&quantized, scale, zero_point, qmin, qmax)?;
                        quantized_data.push(dequantized);

                        data_idx += 1;
                    }
                }
            }
        }

        // Create output tensor with same shape as input
        Tensor::from_vec(quantized_data, input_shape)
            .map_err(|e| NNError::TensorError { source: e })
    }

    /// Quantize a single value
    fn quantize_value(
        &self,
        val: &T,
        scale: &T,
        zero_point: &T,
        qmin: i32,
        qmax: i32,
    ) -> Result<i32>
    where
        T: Clone + PartialOrd,
    {
        let quantized = <T as QuantizationOps<T>>::quantize_value(
            val,
            scale,
            zero_point,
            self.scheme,
            BITS as usize,
        )?;
        Ok(quantized.max(qmin).min(qmax))
    }

    /// Dequantize a single quantized value
    fn dequantize_value(
        &self,
        quantized: &i32,
        scale: &T,
        zero_point: &T,
        _qmin: i32,
        _qmax: i32,
    ) -> Result<T>
    where
        T: Clone,
    {
        <T as QuantizationOps<T>>::dequantize_value(quantized, scale, zero_point, self.scheme)
    }

    /// Returns quantization range based on bitwidth
    ///
    /// # Returns
    /// (min, max) quantization values
    const fn quantization_range() -> (i32, i32) {
        <T as QuantizationOps<T>>::quantization_range(BITS)
    }

    /// Update quantization parameters from observed tensor statistics
    ///
    /// Computes optimal scale and zero_point based on tensor min/max values:
    /// - Per-tensor: single scale/zero_point for entire tensor
    /// - Per-channel: separate scale/zero_point per channel
    /// - scale = (max - min) / (qmax - qmin)
    /// - zero_point = round(qmin - min / scale) for affine quantization
    /// - zero_point = 0 for symmetric quantization
    ///
    /// # Arguments
    /// * `tensor` - Observed tensor for parameter estimation
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn update_params(&mut self, tensor: &Tensor<B, S, T>) -> Result<()>
    where
        T: Clone + PartialOrd,
    {
        let data = tensor.as_slice();
        let shape = tensor.shape().dims();

        if data.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Cannot update quantization parameters from empty tensor".to_string(),
            });
        }

        // Calculate quantization range
        let (qmin, qmax) = Self::quantization_range();

        match self.granularity {
            QuantizationGranularity::PerTensor => self.update_per_tensor_params(data, qmin, qmax),
            QuantizationGranularity::PerChannel => {
                self.update_per_channel_params(data, shape, qmin, qmax)
            }
        }
    }

    /// Update parameters for per-tensor quantization
    fn update_per_tensor_params(&mut self, data: &[T], qmin: i32, qmax: i32) -> Result<()>
    where
        T: Clone + PartialOrd,
    {
        // Find global min and max values without unnecessary cloning
        let mut min_idx = 0;
        let mut max_idx = 0;

        for (i, val) in data.iter().enumerate().skip(1) {
            if *val < data[min_idx] {
                min_idx = i;
            }
            if *val > data[max_idx] {
                max_idx = i;
            }
        }

        let min_val = data[min_idx].clone();
        let max_val = data[max_idx].clone();

        // Calculate scale and zero_point
        let (scale_val, zero_point_val) =
            self.compute_scale_zero_point(&min_val, &max_val, qmin, qmax)?;

        // Update parameters efficiently
        self.scale.data_mut().clear();
        self.scale.data_mut().push(scale_val);

        self.zero_point.data_mut().clear();
        self.zero_point.data_mut().push(zero_point_val);

        Ok(())
    }

    /// Update parameters for per-channel quantization
    fn update_per_channel_params(
        &mut self,
        data: &[T],
        shape: &[usize],
        qmin: i32,
        qmax: i32,
    ) -> Result<()>
    where
        T: Clone + PartialOrd,
    {
        // Assume channels are the last dimension
        let channel_dim = shape.len() - 1;
        let channels_per_sample = shape[channel_dim];

        if channels_per_sample != self.num_channels {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Tensor has {} channels but quantization expects {}",
                    channels_per_sample, self.num_channels
                ),
            });
        }

        // Compute per-channel statistics
        let mut scale_data = Vec::with_capacity(self.num_channels);
        let mut zero_point_data = Vec::with_capacity(self.num_channels);

        // Calculate elements per channel
        let elements_per_channel = data.len() / self.num_channels;

        for channel in 0..self.num_channels {
            // Find min/max for this channel without creating intermediate vec
            let mut min_idx = channel;
            let mut max_idx = channel;

            for i in 1..elements_per_channel {
                let idx = i * self.num_channels + channel;
                if idx >= data.len() {
                    break;
                }

                if data[idx] < data[min_idx] {
                    min_idx = idx;
                }
                if data[idx] > data[max_idx] {
                    max_idx = idx;
                }
            }

            let min_val = data[min_idx].clone();
            let max_val = data[max_idx].clone();

            // Calculate scale and zero_point for this channel
            let (scale_val, zero_point_val) =
                self.compute_scale_zero_point(&min_val, &max_val, qmin, qmax)?;
            scale_data.push(scale_val);
            zero_point_data.push(zero_point_val);
        }

        // Update parameters efficiently
        *self.scale.data_mut() = scale_data;
        *self.zero_point.data_mut() = zero_point_data;

        Ok(())
    }

    /// Compute scale and zero_point from min/max values
    fn compute_scale_zero_point(
        &self,
        min_val: &T,
        max_val: &T,
        qmin: i32,
        qmax: i32,
    ) -> Result<(T, T)>
    where
        T: Clone + PartialOrd,
    {
        let min_f = <T as QuantizationOps<T>>::to_f64_checked(min_val)?;
        let max_f = <T as QuantizationOps<T>>::to_f64_checked(max_val)?;

        let qmin_f = f64::from(qmin);
        let qmax_f = f64::from(qmax);
        let qrange_f = qmax_f - qmin_f;

        if !qrange_f.is_finite() || qrange_f == 0.0 {
            return Err(NNError::NumericalError {
                message: "Invalid quantization range".to_string(),
            });
        }

        let scale_f = if max_f == min_f {
            1.0
        } else {
            (max_f - min_f) / qrange_f
        };

        if !scale_f.is_finite() || scale_f == 0.0 {
            return Err(NNError::NumericalError {
                message: "Invalid quantization scale".to_string(),
            });
        }

        let scale_val = <T as QuantizationOps<T>>::from_f64_checked(scale_f)?;

        let zero_point_val = match self.scheme {
            QuantizationScheme::Affine => {
                let zp_f = (qmin_f - min_f / scale_f).round().max(qmin_f).min(qmax_f);
                <T as QuantizationOps<T>>::from_f64_checked(zp_f)?
            }
            QuantizationScheme::Symmetric => T::zero(),
        };

        Ok((scale_val, zero_point_val))
    }
}

impl<B, S, T, const BITS: usize> Module<B, S, T> for FakeQuantize<B, S, T, BITS>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType
        + num_traits::Float
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + Clone
        + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: Clone + PartialOrd,
    {
        // Get scale and zero_point from parameters
        let scale_data = self.scale.data().as_slice();
        let zero_point_data = self.zero_point.data().as_slice();

        self.fake_quantize(input, scale_data, zero_point_data)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.scale.clone(), self.zero_point.clone()]
    }

    fn zero_grad(&mut self) {
        self.scale.zero_grad();
        self.zero_point.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.scale.train(mode);
        self.zero_point.train(mode);
    }

    fn name(&self) -> &str {
        "FakeQuantize"
    }
}

// ModuleSerialize is automatically implemented via blanket implementation
