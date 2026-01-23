//! Quantization utilities and calibration data

use super::params::{QuantizationParams, QuantizationError, QuantizationNoiseAnalysis, QuantizationResult};
use serde::{Deserialize, Serialize};

/// Dynamic quantization using min-max range estimation
pub struct MinMaxQuantizer;

impl MinMaxQuantizer {
    /// Compute quantization parameters from a slice of floating-point values
    ///
    /// # Arguments
    /// * `data` - The floating-point data to analyze
    /// * `num_bits` - Number of bits for quantization (8 for QInt8/QUInt8)
    /// * `signed` - Whether to use signed quantization (`QInt8`) or unsigned (`QUInt8`)
    ///
    /// # Returns
    /// Optimal quantization parameters
    ///
    /// # Panics
    /// Panics if `data` is empty
    #[must_use]
    pub fn compute_params(data: &[f32], num_bits: u32, signed: bool) -> QuantizationParams {
        assert!(!data.is_empty(), "Cannot quantize empty data");

        // Find min and max values
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in data {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        Self::compute_params_from_range(min_val, max_val, num_bits, signed)
    }

    /// Compute quantization parameters from known min/max range
    ///
    /// # Arguments
    /// * `min_val` - Minimum value in the data
    /// * `max_val` - Maximum value in the data
    /// * `num_bits` - Number of bits for quantization
    /// * `signed` - Whether to use signed quantization
    ///
    /// # Returns
    /// Quantization parameters
    #[must_use]
    pub fn compute_params_from_range(
        min_val: f32,
        max_val: f32,
        num_bits: u32,
        signed: bool,
    ) -> QuantizationParams {
        let qmin = if signed {
            -(2_i32.pow(num_bits - 1))
        } else {
            0
        };
        let qmax = if signed {
            2_i32.pow(num_bits - 1) - 1
        } else {
            2_i32.pow(num_bits) - 1
        };

        // Compute scale: (max_val - min_val) / (qmax - qmin)
        let scale = if (max_val - min_val).abs() < f32::EPSILON {
            1.0 // Avoid division by zero for constant tensors
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                (max_val - min_val) / (qmax - qmin) as f32
            }
        };

        // For simplicity, use zero_point = 0 (symmetric around zero)
        // This is common in many quantization implementations
        let zero_point = 0;

        QuantizationParams { scale, zero_point }
    }

    /// Analyze quantization error for given parameters
    ///
    /// # Arguments
    /// * `data` - The original floating-point data
    /// * `params` - Quantization parameters to analyze
    ///
    /// # Returns
    /// Quantization error metrics
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_error(data: &[f32], params: &QuantizationParams) -> QuantizationError {
        let result = Self::quantize_data(data, params, true);
        let mut errors = Vec::new();
        let mut max_error = 0.0_f32;
        let mut mse = 0.0_f32;

        for (i, &original) in data.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let quantized_val = f32::from(result.data[i]) * params.scale + params.zero_point as f32;
            let error = (original - quantized_val).abs();
            errors.push(error);
            max_error = max_error.max(error);
            mse += error * error;
        }

        mse /= data.len() as f32;

        QuantizationError {
            max_error,
            mse,
            mean_error: errors.iter().sum::<f32>() / errors.len() as f32,
            errors,
        }
    }

    /// Perform comprehensive quantization noise analysis
    ///
    /// # Arguments
    /// * `data` - Original floating-point data
    /// * `params` - Quantization parameters used
    ///
    /// # Returns
    /// Detailed noise analysis including SNR, PSNR, and error distribution
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn analyze_noise(data: &[f32], params: &QuantizationParams) -> QuantizationNoiseAnalysis {
        let error_stats = Self::analyze_error(data, params);

        // Calculate signal power (variance of original data)
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let signal_power =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        // Noise power is the MSE
        let noise_power = error_stats.mse;
        let noise_variance = error_stats
            .errors
            .iter()
            .map(|&e| (e - error_stats.mean_error).powi(2))
            .sum::<f32>()
            / error_stats.errors.len() as f32;

        // Calculate SNR and PSNR
        let snr_db = if noise_power > 0.0 && signal_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        let max_signal = data.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let psnr_db = if error_stats.max_error > 0.0 && max_signal > 0.0 {
            20.0 * (max_signal / error_stats.max_error).log10()
        } else {
            f32::INFINITY
        };

        // Calculate Effective Number of Bits (ENOB)
        // ENOB = (SNR - 1.76) / 6.02 for sinusoidal signals
        // Using a simplified approximation for general signals
        let enob = if snr_db > 0.0 {
            (snr_db - 1.76) / 6.02
        } else {
            0.0
        };

        // Create error distribution histogram
        let max_error = error_stats.max_error;
        let mut error_histogram = [0usize; 10];

        for &error in &error_stats.errors {
            let bin = if max_error > 0.0 {
                ((error / max_error) * 9.0).floor() as usize
            } else {
                0
            };
            let bin = bin.min(9);
            error_histogram[bin] += 1;
        }

        QuantizationNoiseAnalysis {
            error_stats,
            snr_db,
            psnr_db,
            noise_variance,
            signal_power,
            noise_power,
            quantization_step: params.scale,
            enob: enob.max(0.0),
            error_histogram,
        }
    }

    /// Quantize a slice of floating-point data using computed parameters
    ///
    /// # Arguments
    /// * `data` - The floating-point data to quantize
    /// * `params` - Quantization parameters to use
    /// * `signed` - Whether to use signed quantization
    ///
    /// # Returns
    /// Quantized data as `QInt8` or `QUInt8` values
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn quantize_data(
        data: &[f32],
        params: &QuantizationParams,
        signed: bool,
    ) -> QuantizationResult<i8> {
        let quantized_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                // q = round((x - zero_point) / scale)
                let quantized = ((x - params.zero_point as f32) / params.scale).round();
                if signed {
                    quantized.clamp(f32::from(i8::MIN), f32::from(i8::MAX)) as i8
                } else {
                    quantized.clamp(f32::from(u8::MIN), f32::from(u8::MAX)) as i8
                }
            })
            .collect();

        QuantizationResult {
            data: quantized_data,
            params: *params,
        }
    }
}

/// Symmetric quantization utilities
pub struct SymmetricQuantizer;

impl SymmetricQuantizer {
    /// Compute symmetric quantization parameters (`zero_point` = 0)
    ///
    /// # Arguments
    /// * `data` - The floating-point data to analyze
    ///
    /// # Returns
    /// Symmetric quantization parameters
    /// # Panics
    /// Panics if `data` is empty
    #[must_use]
    pub fn compute_params(data: &[f32]) -> QuantizationParams {
        assert!(!data.is_empty(), "Cannot quantize empty data");

        // For symmetric quantization, find the absolute maximum
        let mut abs_max = 0.0_f32;
        for &val in data {
            let abs_val = val.abs();
            if abs_val > abs_max {
                abs_max = abs_val;
            }
        }

        // Scale = abs_max / (2^(bits-1) - 1) for signed 8-bit
        let scale = if abs_max.abs() < f32::EPSILON {
            1.0
        } else {
            abs_max / 127.0 // For signed 8-bit: -127 to +127
        };

        QuantizationParams {
            scale,
            zero_point: 0, // Symmetric quantization always uses zero_point = 0
        }
    }
}

/// Percentile-based quantization for robust parameter estimation
pub struct PercentileQuantizer;

impl PercentileQuantizer {
    /// Compute quantization parameters using percentile-based range estimation
    ///
    /// This method is more robust to outliers than min-max quantization.
    ///
    /// # Arguments
    /// * `data` - The floating-point data to analyze
    /// * `lower_percentile` - Lower percentile for range estimation (e.g., 0.01 for 1%)
    /// * `upper_percentile` - Upper percentile for range estimation (e.g., 0.99 for 99%)
    /// * `signed` - Whether to use signed quantization
    ///
    /// # Returns
    /// Quantization parameters
    /// # Panics
    /// Panics if `data` is empty or percentiles are invalid
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn compute_params(
        data: &[f32],
        lower_percentile: f32,
        upper_percentile: f32,
        signed: bool,
    ) -> QuantizationParams {
        assert!(!data.is_empty(), "Cannot quantize empty data");
        assert!(
            (0.0..=1.0).contains(&lower_percentile),
            "Lower percentile must be between 0 and 1"
        );
        assert!(
            (0.0..=1.0).contains(&upper_percentile),
            "Upper percentile must be between 0 and 1"
        );
        assert!(
            lower_percentile < upper_percentile,
            "Lower percentile must be less than upper percentile"
        );

        // Sort data to compute percentiles
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = (lower_percentile * (sorted_data.len() - 1) as f32) as usize;
        let upper_idx = (upper_percentile * (sorted_data.len() - 1) as f32) as usize;

        let min_val = sorted_data[lower_idx];
        let max_val = sorted_data[upper_idx];

        MinMaxQuantizer::compute_params_from_range(min_val, max_val, 8, signed)
    }
}

/// Quantization calibration data for static quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Minimum values observed during calibration
    pub min_values: Vec<f32>,
    /// Maximum values observed during calibration
    pub max_values: Vec<f32>,
}

impl CalibrationData {
    /// Create new calibration data
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_values: Vec::new(),
            max_values: Vec::new(),
        }
    }
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationData {
    /// Update calibration data with new tensor values
    ///
    /// # Panics
    /// Panics if data is empty (unwraps on min/max operations)
    pub fn update(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let min_val = data
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_val = data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        self.min_values.push(min_val);
        self.max_values.push(max_val);
    }

    /// Compute final quantization parameters from calibration data
    ///
    /// # Arguments
    /// * `signed` - Whether to use signed quantization
    ///
    /// # Returns
    /// Quantization parameters based on observed ranges
    /// # Panics
    /// Panics if no calibration data is available
    #[must_use]
    pub fn compute_params(&self, signed: bool) -> QuantizationParams {
        assert!(!self.min_values.is_empty(), "No calibration data available");

        let overall_min = self
            .min_values
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let overall_max = self
            .max_values
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        MinMaxQuantizer::compute_params_from_range(overall_min, overall_max, 8, signed)
    }
}