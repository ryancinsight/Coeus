//! Serialization support for quantized models

use crate::core::error::Result;
use quantization::{
    CalibrationPipeline, SerializableCalibrationPipeline,
    CalibrationConfig, MixedPrecisionConfig, QuantizationScheme, QuantizedWeights,
};

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

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

/// Serializable representation of mixed precision quantized linear layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMixedPrecisionQuantizedLinear {
    /// Quantized weights in serializable format
    pub weight: SerializableQuantizedWeights,
    /// Weight quantization scale (as f64)
    pub weight_scale: f64,
    /// Weight quantization zero point (as f64)
    pub weight_zero_point: f64,
    /// Bias tensor data (optional, flattened)
    pub bias: Option<(Vec<f64>, Vec<usize>)>,
    /// Input quantization scale (as f64)
    pub input_scale: f64,
    /// Input quantization zero point (as f64)
    pub input_zero_point: f64,
    /// Output quantization scale (as f64)
    pub output_scale: f64,
    /// Output quantization zero point (as f64)
    pub output_zero_point: f64,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Layer name for mixed precision configuration
    pub layer_name: String,
}

/// Comprehensive mixed precision model serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionModelArchive {
    /// Mixed precision configuration
    pub config: MixedPrecisionConfig,
    /// Calibration pipeline state
    pub calibration: Option<SerializableCalibrationPipeline>,
    /// Layer configurations (layer_name -> serializable layer)
    pub layers: std::collections::HashMap<String, SerializableMixedPrecisionQuantizedLinear>,
    /// Model metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Trait for mixed precision model serialization
pub trait MixedPrecisionSerialize<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + PartialOrd + Into<f64> + From<f64>,
    f64: From<T>,
{
    /// Save mixed precision model to a comprehensive archive
    fn save_mixed_precision(&self, path: &std::path::Path) -> Result<()>;

    /// Load mixed precision model from archive
    fn load_mixed_precision(path: &std::path::Path, backend: B) -> Result<Self>
    where
        Self: Sized;
}
