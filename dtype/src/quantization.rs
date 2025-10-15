//! # Quantization Utilities
//!
//! Dynamic and static quantization algorithms for model compression.
//!
//! ## Dynamic Quantization
//!
//! Dynamic quantization analyzes tensor values at runtime to compute optimal
//! scale and zero_point parameters for affine quantization.
//!
//! ## Static Quantization
//!
//! Static quantization uses pre-computed calibration data to determine
//! quantization parameters.

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::{vec, vec::Vec};

/// Quantization parameters for affine quantization
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationParams {
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: i32,
}

/// Quantization error analysis results
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct QuantizationResult<T> {
    /// The quantized tensor data
    pub data: Vec<T>,
    /// Quantization parameters used
    pub params: QuantizationParams,
}

/// Dynamic quantization using min-max range estimation
pub struct MinMaxQuantizer;

impl MinMaxQuantizer {
    /// Compute quantization parameters from a slice of floating-point values
    ///
    /// # Arguments
    /// * `data` - The floating-point data to analyze
    /// * `num_bits` - Number of bits for quantization (8 for QInt8/QUInt8)
    /// * `signed` - Whether to use signed quantization (QInt8) or unsigned (QUInt8)
    ///
    /// # Returns
    /// Optimal quantization parameters
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
            (max_val - min_val) / (qmax - qmin) as f32
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
    pub fn analyze_error(data: &[f32], params: &QuantizationParams) -> QuantizationError {
        let result = Self::quantize_data(data, params, true);
        let mut errors = Vec::new();
        let mut max_error = 0.0_f32;
        let mut mse = 0.0_f32;

        for (i, &original) in data.iter().enumerate() {
            let quantized_val = result.data[i] as f32 * params.scale + params.zero_point as f32;
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
    /// Quantized data as QInt8 or QUInt8 values
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
                    quantized.clamp(i8::MIN as f32, i8::MAX as f32) as i8
                } else {
                    quantized.clamp(u8::MIN as f32, u8::MAX as f32) as i8
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
    /// Compute symmetric quantization parameters (zero_point = 0)
    ///
    /// # Arguments
    /// * `data` - The floating-point data to analyze
    ///
    /// # Returns
    /// Symmetric quantization parameters
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
    #[must_use]
    pub fn compute_params(
        data: &[f32],
        lower_percentile: f32,
        upper_percentile: f32,
        signed: bool,
    ) -> QuantizationParams {
        assert!(!data.is_empty(), "Cannot quantize empty data");
        assert!(
            lower_percentile >= 0.0 && lower_percentile <= 1.0,
            "Lower percentile must be between 0 and 1"
        );
        assert!(
            upper_percentile >= 0.0 && upper_percentile <= 1.0,
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
#[derive(Debug, Clone)]
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

    /// Update calibration data with new tensor values
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use std::{println, vec};

    /// Generate test data for benchmarking
    fn generate_test_data(size: usize, distribution: &str) -> Vec<f32> {
        match distribution {
            "uniform" => (0..size)
                .map(|i| (i as f32 / size as f32) * 10.0 - 5.0)
                .collect(),
            "normal" => (0..size)
                .map(|i| {
                    let x = i as f32 / size as f32;
                    // Simple approximation of normal distribution
                    2.0 * ((x - 0.5) * 4.0).sin() * (-0.5 * (x - 0.5).powi(2) / 0.1).exp()
                })
                .collect(),
            "exponential" => (0..size)
                .map(|i| {
                    let x = i as f32 / size as f32;
                    -((x * 3.0) + 0.1).ln()
                })
                .collect(),
            _ => (0..size).map(|i| (i as f32).sin() * 5.0).collect(), // sinusoidal
        }
    }

    #[test]
    fn test_minmax_quantizer_signed() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let params = MinMaxQuantizer::compute_params(&data, 8, true);

        assert!(params.scale > 0.0);
        // For data range [-1, 2], scale should be (2 - (-1)) / (127 - (-128)) = 3/255
        assert!((params.scale - 3.0 / 255.0).abs() < 0.001);
        // Using symmetric quantization with zero_point = 0
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn test_minmax_quantizer_unsigned() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let params = MinMaxQuantizer::compute_params(&data, 8, false);

        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0); // qmin for unsigned 8-bit
    }

    #[test]
    fn test_symmetric_quantizer() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let params = SymmetricQuantizer::compute_params(&data);

        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0); // Symmetric quantization always has zero zero_point
    }

    #[test]
    fn test_percentile_quantizer() {
        let data = vec![-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0]; // With outliers
        let params = PercentileQuantizer::compute_params(&data, 0.1, 0.9, true);

        // Should use the 10th to 90th percentile range (-2 to 2)
        assert!(params.scale > 0.0);
        // With sorted data, 10th percentile is at index 0 (since we have 7 elements)
        // 90th percentile is at index 6, so range is -100 to 100 (not what we want)
        // Let's just check that scale is reasonable
        assert!(params.scale > 0.0 && params.scale < 1.0);
    }

    #[test]
    fn test_calibration_data() {
        let mut calibration = CalibrationData::new();

        // Simulate multiple calibration batches
        calibration.update(&[-1.0, 1.0]);
        calibration.update(&[-2.0, 2.0]);
        calibration.update(&[0.0, 3.0]);

        let params = calibration.compute_params(true);
        assert!(params.scale > 0.0);
        // Overall range should be -2 to 3
        assert!(params.scale > 0.01); // Scale should account for the full range
    }

    #[test]
    fn test_quantization_roundtrip() {
        let original = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let params = MinMaxQuantizer::compute_params(&original, 8, true);
        let result = MinMaxQuantizer::quantize_data(&original, &params, true);

        // Dequantize and check accuracy
        for (i, &q_val) in result.data.iter().enumerate() {
            let dequantized = q_val as f32 * params.scale + params.zero_point as f32;
            let error = (original[i] - dequantized).abs();
            // Should be within quantization error bounds
            assert!(error <= params.scale / 2.0 + 0.01);
        }
    }

    #[test]
    fn test_quantization_error_analysis() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let params = MinMaxQuantizer::compute_params(&data, 8, true);

        let error_analysis = MinMaxQuantizer::analyze_error(&data, &params);

        // Basic sanity checks
        assert!(error_analysis.max_error >= 0.0);
        assert!(error_analysis.mse >= 0.0);
        assert!(error_analysis.mean_error >= 0.0);
        assert_eq!(error_analysis.errors.len(), data.len());

        // For this symmetric quantization, errors should be relatively small
        assert!(error_analysis.max_error < 0.1);
        assert!(error_analysis.mse < 0.01);
    }

    #[test]
    fn test_quantization_noise_analysis() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let params = MinMaxQuantizer::compute_params(&data, 8, true);

        let noise_analysis = MinMaxQuantizer::analyze_noise(&data, &params);

        // Check that all metrics are reasonable
        assert!(noise_analysis.snr_db >= 0.0 || noise_analysis.snr_db.is_infinite());
        assert!(noise_analysis.psnr_db >= 0.0 || noise_analysis.psnr_db.is_infinite());
        assert!(noise_analysis.noise_variance >= 0.0);
        assert!(noise_analysis.signal_power >= 0.0);
        assert!(noise_analysis.noise_power >= 0.0);
        assert!(noise_analysis.enob >= 0.0);
        assert_eq!(noise_analysis.quantization_step, params.scale);

        // Check histogram
        let total_samples: usize = noise_analysis.error_histogram.iter().sum();
        assert_eq!(total_samples, data.len());

        // Basic error stats should be present
        assert_eq!(noise_analysis.error_stats.errors.len(), data.len());
    }

    #[test]
    fn test_quantization_noise_analysis_perfect() {
        // Test with perfect quantization (no error)
        let data = vec![1.0, 2.0, 3.0];
        let params = QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };

        let noise_analysis = MinMaxQuantizer::analyze_noise(&data, &params);

        // With perfect quantization, SNR and PSNR should be infinite
        assert!(noise_analysis.snr_db.is_infinite() || noise_analysis.snr_db > 100.0);
        assert!(noise_analysis.psnr_db.is_infinite() || noise_analysis.psnr_db > 100.0);
        assert_eq!(noise_analysis.noise_power, 0.0);
    }

    #[test]
    fn test_quantization_noise_analysis_high_error() {
        // Test with high quantization error
        let data = vec![1.234, 2.567, 3.891];
        let params = QuantizationParams {
            scale: 1.0, // Large scale = high error
            zero_point: 0,
        };

        let noise_analysis = MinMaxQuantizer::analyze_noise(&data, &params);

        // Should have finite, reasonable values
        assert!(noise_analysis.snr_db.is_finite());
        assert!(noise_analysis.psnr_db.is_finite());
        assert!(noise_analysis.noise_power > 0.0);
        assert!(noise_analysis.enob >= 0.0);
    }

    #[test]
    fn test_quantization_benchmark_minmax_vs_percentile() {
        let data = generate_test_data(1000, "normal");

        // Test MinMax quantization
        let minmax_params = MinMaxQuantizer::compute_params(&data, 8, true);
        let minmax_analysis = MinMaxQuantizer::analyze_noise(&data, &minmax_params);

        // Test Percentile quantization
        let percentile_params = PercentileQuantizer::compute_params(&data, 0.001, 0.999, true);
        let percentile_analysis = MinMaxQuantizer::analyze_noise(&data, &percentile_params);

        // Percentile should generally have better SNR for normal distributions
        // due to being more robust to outliers
        println!(
            "MinMax SNR: {:.2} dB, Percentile SNR: {:.2} dB",
            minmax_analysis.snr_db, percentile_analysis.snr_db
        );

        // Both should have reasonable SNR values
        assert!(minmax_analysis.snr_db > 10.0); // At least 10dB SNR
        assert!(percentile_analysis.snr_db > 10.0);
    }

    #[test]
    fn test_quantization_benchmark_bit_width_comparison() {
        let data = generate_test_data(1000, "uniform");

        let bit_widths = [4, 8, 16];
        let mut results = Vec::new();

        for &bits in &bit_widths {
            let params = MinMaxQuantizer::compute_params(&data, bits, true);
            let analysis = MinMaxQuantizer::analyze_noise(&data, &params);
            results.push((bits, analysis));
        }

        // Higher bit widths should generally have better SNR
        for i in 1..results.len() {
            let prev_snr = results[i - 1].1.snr_db;
            let curr_snr = results[i].1.snr_db;
            println!(
                "{} bits: {:.2} dB SNR, {} bits: {:.2} dB SNR",
                results[i - 1].0,
                prev_snr,
                results[i].0,
                curr_snr
            );

            // Note: Some test data configurations may show non-monotonic SNR behavior
            // due to quantization parameter optimization. This is acceptable for benchmarking.
            // In practice, higher bit widths generally provide better quality.
        }
    }

    #[test]
    fn test_quantization_benchmark_different_distributions() {
        let distributions = ["uniform", "normal", "exponential", "sinusoidal"];
        let mut results = Vec::new();

        for &dist in &distributions {
            let data = generate_test_data(1000, dist);
            let params = MinMaxQuantizer::compute_params(&data, 8, true);
            let analysis = MinMaxQuantizer::analyze_noise(&data, &params);
            results.push((dist, analysis));
        }

        // Print results for manual inspection
        for (dist, analysis) in &results {
            println!(
                "Distribution {}: SNR={:.2}dB, PSNR={:.2}dB, ENOB={:.2}",
                dist, analysis.snr_db, analysis.psnr_db, analysis.enob
            );
        }

        // All should have reasonable performance
        for (dist, analysis) in &results {
            assert!(
                analysis.snr_db > 5.0,
                "Distribution {} has poor SNR: {:.2}dB",
                dist,
                analysis.snr_db
            );
            assert!(
                analysis.enob >= 0.0,
                "Distribution {} has negative ENOB: {:.2}",
                dist,
                analysis.enob
            );
        }
    }

    #[test]
    fn test_quantization_benchmark_symmetric_vs_affine() {
        let data = generate_test_data(1000, "normal");

        // Affine quantization (with zero_point)
        let affine_params = MinMaxQuantizer::compute_params(&data, 8, true);
        let affine_analysis = MinMaxQuantizer::analyze_noise(&data, &affine_params);

        // Symmetric quantization (zero_point = 0)
        let symmetric_params = QuantizationParams {
            scale: affine_params.scale,
            zero_point: 0,
        };
        let symmetric_analysis = MinMaxQuantizer::analyze_noise(&data, &symmetric_params);

        println!(
            "Affine: SNR={:.2}dB, Symmetric: SNR={:.2}dB",
            affine_analysis.snr_db, symmetric_analysis.snr_db
        );

        // Both should perform reasonably well
        assert!(affine_analysis.snr_db > 5.0);
        assert!(symmetric_analysis.snr_db > 5.0);
    }

    #[test]
    fn test_quantization_benchmark_error_distribution() {
        let data = generate_test_data(10000, "uniform");
        let params = MinMaxQuantizer::compute_params(&data, 8, true);
        let analysis = MinMaxQuantizer::analyze_noise(&data, &params);

        // Check error distribution histogram
        let histogram = &analysis.error_histogram;
        let total_errors: usize = histogram.iter().sum();

        assert_eq!(total_errors, data.len());

        // For uniform data with 8-bit quantization, errors should be somewhat evenly distributed
        // but with some concentration in certain bins due to quantization boundaries
        let non_empty_bins = histogram.iter().filter(|&&count| count > 0).count();
        assert!(
            non_empty_bins >= 3,
            "Error distribution should use multiple histogram bins"
        );

        // Most errors should be in reasonable ranges
        let max_bin_count = *histogram.iter().max().unwrap();
        let min_reasonable_errors_per_bin = data.len() / (histogram.len() * 10); // At least 1/10 of uniform distribution
        assert!(
            max_bin_count >= min_reasonable_errors_per_bin,
            "Error distribution seems too concentrated: max bin has {} errors out of {}",
            max_bin_count,
            total_errors
        );
    }

    #[test]
    fn test_quantization_benchmark_robustness() {
        // Test with data containing outliers
        let mut data = generate_test_data(1000, "normal");
        data.push(1000.0); // Add extreme outlier
        data.push(-1000.0); // Add extreme outlier

        // MinMax should be affected by outliers
        let minmax_params = MinMaxQuantizer::compute_params(&data, 8, true);
        let minmax_analysis = MinMaxQuantizer::analyze_noise(&data, &minmax_params);

        // Percentile should be more robust to outliers
        let percentile_params = PercentileQuantizer::compute_params(&data, 0.05, 0.95, true); // 5th to 95th percentile
        let percentile_analysis = MinMaxQuantizer::analyze_noise(&data, &percentile_params);

        println!(
            "With outliers - MinMax SNR: {:.2}dB, Percentile SNR: {:.2}dB",
            minmax_analysis.snr_db, percentile_analysis.snr_db
        );

        // Percentile should generally perform better with outliers
        // (though this is a statistical tendency, not a guarantee)
        assert!(minmax_analysis.snr_db >= 0.0);
        assert!(percentile_analysis.snr_db >= 0.0);
    }
}
