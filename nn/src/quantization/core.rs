//! Core quantization types and enums

use crate::error::{NNError, Result};

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageFromVec, QuantizedStorage, QuantizedStorage4, QuantizedStorage8, QuantizedStorage16};
use coeus_tensor::Tensor;

use serde::{Deserialize, Serialize};

/// Serializable representation of quantized weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableQuantizedWeights {
    /// Bitwidth of the quantized weights
    pub bitwidth: QuantizationBitwidth,
    /// Shape of the weight tensor
    pub shape: Vec<usize>,
    /// Flattened quantized weight data
    pub data: Vec<f64>, // Store as f64 for serialization
    /// Quantization scale
    pub scale: f64,
    /// Quantization zero point
    pub zero_point: f64,
}

/// Enum for different quantized weight storage types
#[derive(Debug)]
pub enum QuantizedWeights<B, S, T>
where
    B: Backend,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// 4-bit quantized weights
    Bits4(Tensor<B, QuantizedStorage<T, 4>, T>),
    /// 8-bit quantized weights
    Bits8(Tensor<B, QuantizedStorage<T, 8>, T>),
    /// 16-bit quantized weights
    Bits16(Tensor<B, QuantizedStorage<T, 16>, T>),
}

impl<B, S, T> QuantizedWeights<B, S, T>
where
    B: Backend,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Get the bitwidth of the weights
    #[must_use]
    pub fn bitwidth(&self) -> usize {
        match self {
            Self::Bits4(_) => 4,
            Self::Bits8(_) => 8,
            Self::Bits16(_) => 16,
        }
    }

    /// Get a reference to the underlying tensor as a trait object
    #[must_use]
    pub fn as_storage_ref(&self) -> &dyn Storage<T> {
        match self {
            Self::Bits4(tensor) => tensor.storage_ref(),
            Self::Bits8(tensor) => tensor.storage_ref(),
            Self::Bits16(tensor) => tensor.storage_ref(),
        }
    }

    /// Convert to serializable representation
    #[must_use]
    pub fn to_serializable(&self) -> SerializableQuantizedWeights
    where
        T: Into<f64> + Clone,
        f64: From<T>,
    {
        match self {
            Self::Bits4(tensor) => Self::tensor_to_serializable(tensor, QuantizationBitwidth::Bits4),
            Self::Bits8(tensor) => Self::tensor_to_serializable(tensor, QuantizationBitwidth::Bits8),
            Self::Bits16(tensor) => Self::tensor_to_serializable(tensor, QuantizationBitwidth::Bits16),
        }
    }

    /// Helper function to convert tensor to serializable format
    fn tensor_to_serializable<const BITS: usize>(
        tensor: &Tensor<B, QuantizedStorage<T, BITS>, T>,
        bitwidth: QuantizationBitwidth,
    ) -> SerializableQuantizedWeights
    where
        T: Into<f64> + Clone,
        f64: From<T>,
    {
        let shape = tensor.shape().dims();
        let data: Vec<f64> = tensor.as_slice().iter().map(|&x| f64::from(x.clone())).collect();

        // For now, use default scale/zero_point. In a real implementation,
        // these would be extracted from the QuantizedStorage metadata
        let scale = 1.0;
        let zero_point = 0.0;

        SerializableQuantizedWeights {
            bitwidth,
            shape: shape.to_vec(),
            data,
            scale,
            zero_point,
        }
    }

    /// Create from serializable representation
    pub fn from_serializable(
        serializable: SerializableQuantizedWeights,
        backend: B,
    ) -> Result<Self>
    where
        T: From<f64>,
    {
        let data: Vec<T> = serializable.data.into_iter().map(T::from).collect();

        match serializable.bitwidth {
            QuantizationBitwidth::Bits4 => {
                let storage = QuantizedStorage::<T, 4>::from_vec(data, &serializable.shape)?;
                let tensor = Tensor::from_storage(storage, backend);
                Ok(Self::Bits4(tensor))
            }
            QuantizationBitwidth::Bits8 => {
                let storage = QuantizedStorage::<T, 8>::from_vec(data, &serializable.shape)?;
                let tensor = Tensor::from_storage(storage, backend);
                Ok(Self::Bits8(tensor))
            }
            QuantizationBitwidth::Bits16 => {
                let storage = QuantizedStorage::<T, 16>::from_vec(data, &serializable.shape)?;
                let tensor = Tensor::from_storage(storage, backend);
                Ok(Self::Bits16(tensor))
            }
        }
    }
}

/// Quantization scheme enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Affine quantization: q = round((x - zero_point) / scale)
    Affine,
    /// Symmetric quantization: q = round(x / scale)
    Symmetric,
}

/// Granularity of quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Single scale/zero_point for entire tensor
    PerTensor,
    /// Per-channel scale/zero_point (e.g., per output channel in Conv2D)
    PerChannel,
}

/// Supported quantization bitwidths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationBitwidth {
    /// 4-bit quantization
    Bits4,
    /// 8-bit quantization
    Bits8,
    /// 16-bit quantization
    Bits16,
}

impl QuantizationBitwidth {
    /// Get the bitwidth value
    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::Bits4 => 4,
            Self::Bits8 => 8,
            Self::Bits16 => 16,
        }
    }
}

/// Advanced calibration techniques for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Min-max calibration (current implementation)
    MinMax,
    /// Percentile-based calibration (e.g., 99.9th percentile)
    Percentile,
    /// MSE minimization calibration
    MseMinimization,
    /// Entropy minimization calibration
    EntropyMinimization,
}

/// Calibration statistics collected from tensor data
#[derive(Debug, Clone)]
pub struct CalibrationStats<T> {
    /// Minimum value observed
    pub min: T,
    /// Maximum value observed
    pub max: T,
    /// Mean value
    pub mean: T,
    /// Standard deviation
    pub std: T,
    /// Percentiles (0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99)
    pub percentiles: Vec<T>,
    /// Histogram bins and counts
    pub histogram: Option<(Vec<T>, Vec<u64>)>,
}

/// Advanced calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Calibration method to use
    pub method: CalibrationMethod,
    /// Number of calibration samples to collect
    pub num_samples: usize,
    /// Percentile value for percentile calibration (0.0-1.0)
    pub percentile: f64,
    /// Number of histogram bins for entropy minimization
    pub histogram_bins: usize,
    /// Enable histogram collection
    pub collect_histogram: bool,
}

impl<T> CalibrationStats<T>
where
    T: DataType + Clone + PartialOrd + Into<f64> + From<f64>,
    f64: From<T>,
{
    /// Collect calibration statistics from tensor data
    pub fn collect(data: &[T]) -> Result<Self> {
        if data.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Cannot collect calibration statistics from empty data".to_string(),
            });
        }

        // Calculate basic statistics
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_data[0].clone();
        let max = sorted_data[sorted_data.len() - 1].clone();

        // Calculate mean
        let sum: f64 = sorted_data.iter().map(|x| f64::from(x.clone())).sum();
        let mean = T::from(sum / sorted_data.len() as f64);

        // Calculate standard deviation
        let variance: f64 = sorted_data.iter()
            .map(|x| {
                let diff = f64::from(x.clone()) - f64::from(mean.clone());
                diff * diff
            })
            .sum::<f64>() / sorted_data.len() as f64;
        let std = T::from(variance.sqrt());

        // Calculate percentiles
        let percentiles = Self::calculate_percentiles(&sorted_data);

        Ok(Self {
            min,
            max,
            mean,
            std,
            percentiles,
            histogram: None, // Will be calculated separately if needed
        })
    }

    /// Calculate percentile values from sorted data
    fn calculate_percentiles(sorted_data: &[T]) -> Vec<T> {
        let percentiles_to_calculate = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999];

        percentiles_to_calculate.iter().map(|&p| {
            let index = (p * (sorted_data.len() - 1) as f64) as usize;
            sorted_data[index].clone()
        }).collect()
    }

    /// Get optimal scale and zero_point using the specified calibration method
    pub fn get_optimal_params(&self, method: CalibrationMethod, bits: usize) -> (T, T) {
        match method {
            CalibrationMethod::MinMax => {
                Self::min_max_calibration(self.min.clone(), self.max.clone())
            }
            CalibrationMethod::Percentile => {
                // Use 99.9th percentile for outlier robustness
                let percentile_idx = 10; // 99.9th percentile
                let p_value = self.percentiles[percentile_idx].clone();
                Self::percentile_calibration(self.min.clone(), p_value)
            }
            CalibrationMethod::MseMinimization => {
                // Use MSE minimization for optimal quantization range
                Self::mse_minimization_calibration(self.min.clone(), self.max.clone(), bits)
            }
            CalibrationMethod::EntropyMinimization => {
                // Use entropy minimization for information preservation
                Self::entropy_minimization_calibration(self.min.clone(), self.max.clone(), bits)
            }
        }
    }

    /// Min-max calibration (traditional approach)
    fn min_max_calibration(min: T, max: T) -> (T, T) {
        let scale = Self::calculate_scale(min.clone(), max.clone(), 8); // Assume 8 bits for now
        let zero_point = T::zero();
        (scale, zero_point)
    }

    /// Percentile-based calibration
    fn percentile_calibration(min: T, p_value: T) -> (T, T) {
        let scale = Self::calculate_scale(min.clone(), p_value, 8);
        let zero_point = T::zero();
        (scale, zero_point)
    }

    /// MSE minimization calibration
    fn mse_minimization_calibration(min: T, max: T, bits: usize) -> (T, T) {
        // This is a simplified implementation
        // Real MSE minimization would require optimization algorithms
        // For now, use a heuristic approach
        let range = f64::from(max) - f64::from(min);
        let optimal_range = range * 0.95; // Reduce range slightly for better quantization

        let optimal_max = T::from(f64::from(min) + optimal_range);
        let scale = Self::calculate_scale(min, optimal_max, bits);
        let zero_point = T::zero();
        (scale, zero_point)
    }

    /// Entropy minimization calibration
    fn entropy_minimization_calibration(min: T, max: T, bits: usize) -> (T, T) {
        // This is a simplified implementation
        // Real entropy minimization would analyze the data distribution
        // For now, use a conservative approach similar to percentile
        let range = f64::from(max) - f64::from(min);
        let optimal_range = range * 0.90; // More conservative than MSE

        let optimal_max = T::from(f64::from(min) + optimal_range);
        let scale = Self::calculate_scale(min, optimal_max, bits);
        let zero_point = T::zero();
        (scale, zero_point)
    }

    /// Calculate scale factor for quantization
    fn calculate_scale(min: T, max: T, bits: usize) -> T {
        let qmax = (1i64 << bits) - 1;
        let range = f64::from(max) - f64::from(min);
        if range > 0.0 {
            T::from(range / qmax as f64)
        } else {
            T::from(1.0) // Fallback for constant tensors
        }
    }
}

/// Mixed precision configuration for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Default bitwidth for layers not specified in layer_configs
    pub default_bitwidth: QuantizationBitwidth,
    /// Per-layer bitwidth configurations (layer_name -> bitwidth)
    pub layer_configs: std::collections::HashMap<String, QuantizationBitwidth>,
    /// Quantization scheme to use
    pub scheme: QuantizationScheme,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Calibration configuration
    pub calibration: CalibrationConfig,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::MinMax,
            num_samples: 1000,
            percentile: 0.999, // 99.9th percentile
            histogram_bins: 2048,
            collect_histogram: false,
        }
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            default_bitwidth: QuantizationBitwidth::Bits8,
            layer_configs: std::collections::HashMap::new(),
            scheme: QuantizationScheme::Affine,
            granularity: QuantizationGranularity::PerTensor,
            calibration: CalibrationConfig::default(),
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new mixed precision configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default bitwidth
    #[must_use]
    pub fn with_default_bitwidth(mut self, bitwidth: QuantizationBitwidth) -> Self {
        self.default_bitwidth = bitwidth;
        self
    }

    /// Set bitwidth for a specific layer
    pub fn with_layer_bitwidth(mut self, layer_name: &str, bitwidth: QuantizationBitwidth) -> Self {
        self.layer_configs.insert(layer_name.to_string(), bitwidth);
        self
    }

    /// Set quantization scheme
    #[must_use]
    pub fn with_scheme(mut self, scheme: QuantizationScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set quantization granularity
    #[must_use]
    pub fn with_granularity(mut self, granularity: QuantizationGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Set calibration configuration
    #[must_use]
    pub fn with_calibration(mut self, calibration: CalibrationConfig) -> Self {
        self.calibration = calibration;
        self
    }

    /// Get bitwidth for a specific layer
    #[must_use]
    pub fn get_layer_bitwidth(&self, layer_name: &str) -> QuantizationBitwidth {
        self.layer_configs
            .get(layer_name)
            .copied()
            .unwrap_or(self.default_bitwidth)
    }
}
