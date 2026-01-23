//! Quantization parameters and error analysis types

use serde::{Deserialize, Serialize};

/// Quantization parameters for affine quantization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: i32,
}

/// Quantization error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationError {
    /// Maximum absolute quantization error
    pub max_error: f32,
    /// Mean squared quantization error
    pub mse: f32,
    /// Mean absolute quantization error
    pub mean_error: f32,
    /// Individual quantization errors for each data point
    pub errors: Vec<f32>,
}

/// Advanced quantization noise analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationNoiseAnalysis {
    /// Basic error statistics
    pub error_stats: QuantizationError,
    /// Signal-to-Noise Ratio (SNR) in dB
    pub snr_db: f32,
    /// Peak Signal-to-Noise Ratio (PSNR) in dB
    pub psnr_db: f32,
    /// Quantization noise variance
    pub noise_variance: f32,
    /// Signal power (variance of original data)
    pub signal_power: f32,
    /// Noise power (variance of quantization error)
    pub noise_power: f32,
    /// Quantization step size (scale factor)
    pub quantization_step: f32,
    /// Effective number of bits (ENOB)
    pub enob: f32,
    /// Error distribution histogram (10 bins)
    pub error_histogram: [usize; 10],
}

/// Result of quantization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationResult<T> {
    /// The quantized tensor data
    pub data: Vec<T>,
    /// Quantization parameters used
    pub params: QuantizationParams,
}