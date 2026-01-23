//! Fake quantization for linear operations

use coeus_error::{Error, Result};
use coeus_error::StorageError as CoeusStorageError;
use coeus_error::NNError;

use crate::algorithms::core::{QuantizationGranularity, QuantizationScheme};

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Fake quantization operation for linear layers
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
pub struct LinearFakeQuantize<B, S, T, const BITS: usize>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Quantization scale parameter(s) - can be per-tensor or per-channel
    pub scale: S,
    /// Quantization zero point parameter(s) - can be per-tensor or per-channel
    pub zero_point: S,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Number of channels (1 for per-tensor, >1 for per-channel)
    pub num_channels: usize,
    /// Phantom data for backend and storage types
    _phantom: core::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T, const BITS: usize> LinearFakeQuantize<B, S, T, BITS>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + num_traits::Float + num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    /// Create a new fake quantization module for linear operations
    ///
    /// # Arguments
    /// * `scheme` - Quantization scheme (Affine or Symmetric)
    /// * `granularity` - Quantization granularity (PerTensor or PerChannel)
    /// * `num_channels` - Number of channels (1 for per-tensor, >1 for per-channel)
    ///
    /// # Returns
    /// Fake quantization module
    pub fn new(
        scheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        num_channels: usize,
    ) -> Result<Self> {
        // Validate bitwidth at compile time
        if BITS != 4 && BITS != 8 && BITS != 16 {
            return Err(Error::Storage(CoeusStorageError::Quantized(format!("Unsupported bitwidth: {}. Supported: 4, 8, 16", BITS))));
        }

        // Validate parameters
        if num_channels == 0 {
            return Err(Error::NN(NNError::InvalidParameter("num_channels must be > 0".to_string())));
        }

        match granularity {
            QuantizationGranularity::PerTensor => {
                if num_channels != 1 {
                    return Err(Error::NN(NNError::InvalidParameter("PerTensor granularity requires num_channels = 1".to_string())));
                }
            }
            QuantizationGranularity::PerChannel => {
                if num_channels < 2 {
                    return Err(Error::NN(NNError::InvalidParameter("PerChannel granularity requires num_channels >= 2".to_string())));
                }
            }
        }

        // Create parameter tensors based on granularity
        let param_shape = match granularity {
            QuantizationGranularity::PerTensor => vec![1],
            QuantizationGranularity::PerChannel => vec![num_channels],
        };

        let scale = S::from_vec(
            vec![T::one(); param_shape.iter().product()],
            &param_shape,
        ).map_err(|e| Error::Storage(CoeusStorageError::Quantized(format!("{:?}", e))))?;

        let zero_point = S::from_vec(
            vec![T::zero(); param_shape.iter().product()],
            &param_shape,
        ).map_err(|e| Error::Storage(CoeusStorageError::Quantized(format!("{:?}", e))))?;

        Ok(Self {
            scale,
            zero_point,
            scheme,
            granularity,
            num_channels,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Forward pass with fake quantization for linear operations
    ///
    /// # Arguments
    /// * `input` - Input storage to quantize
    ///
    /// # Returns
    /// Fake quantized output storage (still floating-point for gradient flow)
    pub fn forward(&self, input: &S) -> Result<S> {
        let scales = self.scale.as_slice();
        let zero_points = self.zero_point.as_slice();
        self.fake_quantize(input, scales, zero_points)
    }

    /// Apply fake quantization to a tensor for linear operations
    ///
    /// Simulates quantization effects by:
    /// 1. Quantizing using the selected scheme and granularity
    /// 2. Clamping to quantization range [qmin, qmax]
    /// 3. Dequantizing (maintains gradient flow via STE)
    ///
    /// # Arguments
    /// * `x` - Input storage
    /// * `scales` - Quantization scale(s) (per-tensor or per-channel)
    /// * `zero_points` - Quantization zero point(s) (per-tensor or per-channel)
    ///
    /// # Returns
    /// Fake quantized storage (maintains gradient flow via STE)
    fn fake_quantize(
        &self,
        x: &S,
        scales: &[T],
        zero_points: &[T],
    ) -> Result<S>
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
                        self.dequantize_value(&quantized, scale, zero_point)?;
                    quantized_data.push(dequantized);
                }
            }
            QuantizationGranularity::PerChannel => {
                // Per-channel quantization for linear layers
                // Assume channels are the last dimension (features)
                let channel_dim = input_shape.len() - 1;
                let channels_per_sample = input_shape[channel_dim];

                if channels_per_sample != self.num_channels {
                    return Err(Error::NN(NNError::InvalidParameter(format!(
                        "Input has {} channels but quantization expects {}",
                        channels_per_sample, self.num_channels
                    ))));
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
                            self.dequantize_value(&quantized, scale, zero_point)?;
                        quantized_data.push(dequantized);

                        data_idx += 1;
                    }
                }
            }
        }

        // Create output storage with same shape as input
        S::from_vec(quantized_data, input_shape)
            .map_err(|e| Error::Storage(CoeusStorageError::Quantized(format!("Storage creation failed: {:?}", e))))
    }

    /// Quantize a single value for linear operations
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
        let quantized = match self.scheme {
            QuantizationScheme::Affine => {
                // q = round((x - zero_point) / scale)
                let val_f = val.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert value to f64".to_string())))?;
                let scale_f = scale.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert scale to f64".to_string())))?;
                let zero_point_f = zero_point.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert zero_point to f64".to_string())))?;
                
                ((val_f - zero_point_f) / scale_f).round() as i32
            }
            QuantizationScheme::Symmetric => {
                // q = round(x / scale), zero_point = 0
                let val_f = val.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert value to f64".to_string())))?;
                let scale_f = scale.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert scale to f64".to_string())))?;
                
                (val_f / scale_f).round() as i32
            }
        };
        
        Ok(quantized.max(qmin).min(qmax))
    }

    /// Dequantize a single quantized value for linear operations
    fn dequantize_value(
        &self,
        quantized: &i32,
        scale: &T,
        zero_point: &T,
    ) -> Result<T>
    where
        T: Clone,
    {
        let dequantized_f = match self.scheme {
            QuantizationScheme::Affine => {
                // x = q * scale + zero_point
                let scale_f = scale.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert scale to f64".to_string())))?;
                let zero_point_f = zero_point.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert zero_point to f64".to_string())))?;

                (*quantized as f64) * scale_f + zero_point_f
            }
            QuantizationScheme::Symmetric => {
                // x = q * scale, zero_point = 0
                let scale_f = scale.to_f64().ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert scale to f64".to_string())))?;

                (*quantized as f64) * scale_f
            }
        };

        T::from_f64(dequantized_f).ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert dequantized value from f64".to_string())))
    }

    /// Returns quantization range based on bitwidth for linear operations
    ///
    /// # Returns
    /// (min, max) quantization values
    const fn quantization_range() -> (i32, i32) {
        match BITS {
            4 => (-8, 7),   // 4-bit signed: -8 to 7
            8 => (-128, 127), // 8-bit signed: -128 to 127
            16 => (-32768, 32767), // 16-bit signed: -32768 to 32767
            _ => (0, 0), // Should never reach here due to validation
        }
    }
}