//! Fake quantization for convolution operations

use coeus_error::{Error, Result};
use coeus_error::StorageError as CoeusStorageError;
use coeus_error::NNError;

use crate::algorithms::core::{QuantizationGranularity, QuantizationScheme};

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Fake quantization operation for convolution layers
///
/// This module simulates quantization effects during training by applying
/// quantization to inputs during forward pass while maintaining gradient flow.
/// Supports both per-tensor and per-channel quantization optimized for convolution operations.
///
/// # Generic Parameters
/// - `B`: Backend type
/// - `S`: Storage type for parameters
/// - `T`: Data type for parameters and computation
/// - `BITS`: Quantization bitwidth (4, 8, or 16)
#[derive(Debug)]
pub struct ConvFakeQuantize<B, S, T, const BITS: usize>
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
    /// Number of output channels (1 for per-tensor, >1 for per-channel)
    pub out_channels: usize,
    /// Phantom data for backend and storage types
    _phantom: core::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T, const BITS: usize> ConvFakeQuantize<B, S, T, BITS>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + num_traits::Float + num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    /// Create a new fake quantization module for convolution operations
    ///
    /// # Arguments
    /// * `scheme` - Quantization scheme (Affine or Symmetric)
    /// * `granularity` - Quantization granularity (PerTensor or PerChannel)
    /// * `out_channels` - Number of output channels (1 for per-tensor, >1 for per-channel)
    ///
    /// # Returns
    /// Fake quantization module
    pub fn new(
        scheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        out_channels: usize,
    ) -> Result<Self> {
        // Validate bitwidth at compile time
        if BITS != 4 && BITS != 8 && BITS != 16 {
            return Err(Error::Storage(CoeusStorageError::Quantized(format!("Unsupported bitwidth: {}. Supported: 4, 8, 16", BITS))));
        }

        // Validate parameters
        if out_channels == 0 {
            return Err(Error::NN(NNError::InvalidParameter("out_channels must be > 0".to_string())));
        }

        match granularity {
            QuantizationGranularity::PerTensor => {
                if out_channels != 1 {
                    return Err(Error::NN(NNError::InvalidParameter("PerTensor granularity requires out_channels = 1".to_string())));
                }
            }
            QuantizationGranularity::PerChannel => {
                if out_channels < 2 {
                    return Err(Error::NN(NNError::InvalidParameter("PerChannel granularity requires out_channels >= 2".to_string())));
                }
            }
        }

        // Create parameter tensors based on granularity
        let param_shape = match granularity {
            QuantizationGranularity::PerTensor => vec![1],
            QuantizationGranularity::PerChannel => vec![out_channels],
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
            out_channels,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Forward pass with fake quantization for convolution operations
    ///
    /// # Arguments
    /// * `input` - Input storage to quantize (typically NCHW format)
    ///
    /// # Returns
    /// Fake quantized output (still floating-point for gradient flow)
    pub fn forward(&self, input: &S) -> Result<S> {
        let scales = self.scale.as_slice();
        let zero_points = self.zero_point.as_slice();
        self.fake_quantize_conv(input, scales, zero_points)
    }

    /// Apply fake quantization to a tensor for convolution operations
    ///
    /// Simulates quantization effects by:
    /// 1. Quantizing using the selected scheme and granularity
    /// 2. Clamping to quantization range [qmin, qmax]
    /// 3. Dequantizing (maintains gradient flow via STE)
    ///
    /// For convolution, per-channel quantization is applied along the channel dimension (C in NCHW).
    ///
    /// # Arguments
    /// * `x` - Input storage (typically NCHW format)
    /// * `scales` - Quantization scale(s) (per-tensor or per-channel)
    /// * `zero_points` - Quantization zero point(s) (per-tensor or per-channel)
    ///
    /// # Returns
    /// Fake quantized storage (maintains gradient flow via STE)
    fn fake_quantize_conv(
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
                // Per-channel quantization for convolution
                // Assume NCHW format: [batch, channels, height, width]
                if input_shape.len() < 4 {
                    return Err(Error::NN(NNError::InvalidParameter("Convolution input must have at least 4 dimensions (NCHW)".to_string())));
                }

                let batch_size = input_shape[0];
                let channels = input_shape[1];
                let height = input_shape[2];
                let width = input_shape[3];

                if channels != self.out_channels {
                    return Err(Error::NN(NNError::InvalidParameter(format!(
                        "Input has {} channels but quantization expects {}",
                        channels, self.out_channels
                    ))));
                }

                // Apply per-channel quantization in NCHW format
                let channel_size = height * width;
                let mut data_idx = 0;

                for _batch in 0..batch_size {
                    for channel in 0..channels {
                        let scale = &scales[channel];
                        let zero_point = &zero_points[channel];

                        // Process all spatial locations for this channel
                        for _spatial in 0..channel_size {
                            let val = &x_data[data_idx];
                            let quantized = self.quantize_value(val, scale, zero_point, qmin, qmax)?;
                            let dequantized =
                                self.dequantize_value(&quantized, scale, zero_point)?;
                            quantized_data.push(dequantized);
                            data_idx += 1;
                        }
                    }
                }
            }
        }

        // Create output storage with same shape as input
        S::from_vec(quantized_data, input_shape)
            .map_err(|e| Error::Storage(CoeusStorageError::Quantized(format!("Storage creation failed: {:?}", e))))
    }

    /// Quantize a single value for convolution operations
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

    /// Dequantize a single quantized value for convolution operations
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

    /// Returns quantization range based on bitwidth for convolution operations
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