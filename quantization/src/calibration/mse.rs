//! MSE-based calibration for quantization

use coeus_error::{Error, Result};
use coeus_error::NNError;
use dtype::DataType;
use num_traits::Float;

/// MSE-based calibration for quantization parameters
///
/// This method minimizes the Mean Squared Error (MSE) between the original
/// and quantized values to find optimal quantization parameters.
pub struct MseCalibrator<T: DataType> {
    /// Number of candidate thresholds to test
    pub num_candidates: usize,
    /// Collected data samples for calibration
    pub samples: Vec<Vec<T>>,
}

impl<T> MseCalibrator<T>
where
    T: DataType + Float + Clone + PartialOrd,
{
    /// Create a new MSE-based calibrator
    ///
    /// # Arguments
    /// * `num_candidates` - Number of candidate thresholds to test
    pub fn new(num_candidates: usize) -> Self {
        Self {
            num_candidates: num_candidates.max(10), // Minimum 10 candidates
            samples: Vec::new(),
        }
    }

    /// Create a calibrator with default settings
    pub fn new_default() -> Self {
        Self::new(100) // Test 100 candidate thresholds
    }

    /// Add calibration data
    pub fn add_sample(&mut self, data: Vec<T>) {
        self.samples.push(data);
    }

    /// Compute optimal quantization parameters using MSE minimization
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    /// * `signed` - Whether to use signed quantization
    ///
    /// # Returns
    /// Optimal (scale, zero_point) parameters
    pub fn compute_params(&self, bits: usize, signed: bool) -> Result<(T, T)> {
        if self.samples.is_empty() {
            return Err(Error::NN(NNError::InvalidParameter("No calibration samples available".to_string())));
        }

        // Collect all data points
        let mut all_data = Vec::new();
        for sample in &self.samples {
            all_data.extend_from_slice(sample);
        }

        if all_data.is_empty() {
            return Err(Error::NN(NNError::InvalidParameter("No data points in calibration samples".to_string())));
        }

        // Sort data for threshold selection
        all_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = all_data[0];
        let max_val = all_data[all_data.len() - 1];

        // Generate candidate thresholds
        let mut best_threshold = max_val;
        let mut min_mse = T::infinity();

        // Test different thresholds based on percentiles
        for i in 0..self.num_candidates {
            let percentile = 0.8 + (0.2 * i as f64 / self.num_candidates as f64); // 80% to 100%
            let threshold_idx = (percentile * (all_data.len() - 1) as f64) as usize;
            let threshold = all_data[threshold_idx.min(all_data.len() - 1)];

            // Calculate MSE for this threshold
            let mse = self.calculate_mse(&all_data, min_val, threshold, bits, signed)?;

            if mse < min_mse {
                min_mse = mse;
                best_threshold = threshold;
            }
        }

        // Compute final parameters from optimal threshold
        self.compute_params_from_range(min_val, best_threshold, bits, signed)
    }

    /// Calculate MSE for a given threshold
    fn calculate_mse(
        &self,
        data: &[T],
        min_val: T,
        max_val: T,
        bits: usize,
        signed: bool,
    ) -> Result<T> {
        let (scale, zero_point) = self.compute_params_from_range(min_val, max_val, bits, signed)?;

        let qmin = if signed {
            -(1i64 << (bits - 1))
        } else {
            0
        };
        let qmax = if signed {
            (1i64 << (bits - 1)) - 1
        } else {
            (1i64 << bits) - 1
        };

        let mut mse = T::zero();
        let mut count = 0;

        for &value in data {
            // Clamp to quantization range
            let clamped_value = value.min(max_val).max(min_val);

            // Quantize
            let quantized_f = ((clamped_value - zero_point) / scale).round();
            let quantized_i = quantized_f.to_i64().unwrap_or(0).max(qmin).min(qmax);

            // Dequantize
            let dequantized = T::from(quantized_i).unwrap() * scale + zero_point;

            // Calculate squared error
            let error = value - dequantized;
            mse = mse + error * error;
            count += 1;
        }

        if count > 0 {
            mse = mse / T::from(count).unwrap();
        }

        Ok(mse)
    }

    /// Compute quantization parameters from a given range
    fn compute_params_from_range(
        &self,
        min_val: T,
        max_val: T,
        bits: usize,
        signed: bool,
    ) -> Result<(T, T)> {
        let qmin = if signed {
            -(1i64 << (bits - 1))
        } else {
            0
        };
        let qmax = if signed {
            (1i64 << (bits - 1)) - 1
        } else {
            (1i64 << bits) - 1
        };

        // Compute scale: (max_val - min_val) / (qmax - qmin)
        let range = max_val - min_val;
        let qrange = T::from(qmax - qmin).unwrap();
        
        let scale = if range > T::zero() {
            range / qrange
        } else {
            T::one() // Avoid division by zero for constant tensors
        };

        // For simplicity, use zero_point = 0 (symmetric quantization)
        // In a more sophisticated implementation, we could optimize zero_point as well
        let zero_point = T::zero();

        Ok((scale, zero_point))
    }

    /// Get the optimal threshold that minimizes MSE
    pub fn get_optimal_threshold(&self, bits: usize, signed: bool) -> Result<T> {
        if self.samples.is_empty() {
            return Err(Error::NN(NNError::InvalidParameter("No calibration samples available".to_string())));
        }

        // Collect all data points
        let mut all_data = Vec::new();
        for sample in &self.samples {
            all_data.extend_from_slice(sample);
        }

        if all_data.is_empty() {
            return Err(Error::NN(NNError::InvalidParameter("No data points in calibration samples".to_string())));
        }

        // Sort data for threshold selection
        all_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = all_data[0];
        let max_val = all_data[all_data.len() - 1];

        // Find optimal threshold
        let mut best_threshold = max_val;
        let mut min_mse = T::infinity();

        for i in 0..self.num_candidates {
            let percentile = 0.8 + (0.2 * i as f64 / self.num_candidates as f64);
            let threshold_idx = (percentile * (all_data.len() - 1) as f64) as usize;
            let threshold = all_data[threshold_idx.min(all_data.len() - 1)];

            let mse = self.calculate_mse(&all_data, min_val, threshold, bits, signed)?;

            if mse < min_mse {
                min_mse = mse;
                best_threshold = threshold;
            }
        }

        Ok(best_threshold)
    }

    /// Clear all calibration samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Get number of calibration samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Set number of candidate thresholds to test
    pub fn set_num_candidates(&mut self, num_candidates: usize) {
        self.num_candidates = num_candidates.max(10);
    }
}