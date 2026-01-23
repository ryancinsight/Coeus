//! Percentile-based calibration for quantization

use coeus_error::{Error, Result};
use coeus_error::NNError;
use dtype::DataType;
use num_traits::Float;

/// Percentile-based calibration for quantization parameters
///
/// This method uses percentile-based range estimation to determine
/// quantization parameters, which is more robust to outliers than min-max.
pub struct PercentileCalibrator<T: DataType> {
    /// Lower percentile for range estimation (e.g., 0.01 for 1%)
    pub lower_percentile: f64,
    /// Upper percentile for range estimation (e.g., 0.99 for 99%)
    pub upper_percentile: f64,
    /// Collected data samples for calibration
    pub samples: Vec<Vec<T>>,
}

impl<T> PercentileCalibrator<T>
where
    T: DataType + Float + Clone + PartialOrd,
{
    /// Create a new percentile-based calibrator
    ///
    /// # Arguments
    /// * `lower_percentile` - Lower percentile (0.0-1.0)
    /// * `upper_percentile` - Upper percentile (0.0-1.0)
    pub fn new(lower_percentile: f64, upper_percentile: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&lower_percentile) {
            return Err(Error::NN(NNError::InvalidParameter("Lower percentile must be between 0.0 and 1.0".to_string())));
        }
        if !(0.0..=1.0).contains(&upper_percentile) {
            return Err(Error::NN(NNError::InvalidParameter("Upper percentile must be between 0.0 and 1.0".to_string())));
        }
        if lower_percentile >= upper_percentile {
            return Err(Error::NN(NNError::InvalidParameter("Lower percentile must be less than upper percentile".to_string())));
        }

        Ok(Self {
            lower_percentile,
            upper_percentile,
            samples: Vec::new(),
        })
    }

    /// Create a calibrator with common percentile settings
    pub fn new_robust() -> Self {
        Self {
            lower_percentile: 0.001, // 0.1%
            upper_percentile: 0.999, // 99.9%
            samples: Vec::new(),
        }
    }

    /// Create a calibrator with conservative percentile settings
    pub fn new_conservative() -> Self {
        Self {
            lower_percentile: 0.01, // 1%
            upper_percentile: 0.99, // 99%
            samples: Vec::new(),
        }
    }

    /// Add calibration data
    pub fn add_sample(&mut self, data: Vec<T>) {
        self.samples.push(data);
    }

    /// Compute optimal quantization parameters using percentile-based range estimation
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

        // Sort data for percentile calculations
        all_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentile indices
        let lower_idx = (self.lower_percentile * (all_data.len() - 1) as f64) as usize;
        let upper_idx = (self.upper_percentile * (all_data.len() - 1) as f64) as usize;

        let min_val = all_data[lower_idx];
        let max_val = all_data[upper_idx];

        // Compute quantization parameters
        self.compute_params_from_range(min_val, max_val, bits, signed)
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
        // This is common in many quantization implementations
        let zero_point = T::zero();

        Ok((scale, zero_point))
    }

    /// Get percentile values from the collected data
    ///
    /// # Returns
    /// (lower_percentile_value, upper_percentile_value)
    pub fn get_percentile_range(&self) -> Result<(T, T)> {
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

        // Sort data for percentile calculations
        all_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentile indices
        let lower_idx = (self.lower_percentile * (all_data.len() - 1) as f64) as usize;
        let upper_idx = (self.upper_percentile * (all_data.len() - 1) as f64) as usize;

        Ok((all_data[lower_idx], all_data[upper_idx]))
    }

    /// Clear all calibration samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Get number of calibration samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Update percentile settings
    pub fn set_percentiles(&mut self, lower: f64, upper: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&lower) {
            return Err(Error::NN(NNError::InvalidParameter("Lower percentile must be between 0.0 and 1.0".to_string())));
        }
        if !(0.0..=1.0).contains(&upper) {
            return Err(Error::NN(NNError::InvalidParameter("Upper percentile must be between 0.0 and 1.0".to_string())));
        }
        if lower >= upper {
            return Err(Error::NN(NNError::InvalidParameter("Lower percentile must be less than upper percentile".to_string())));
        }

        self.lower_percentile = lower;
        self.upper_percentile = upper;
        Ok(())
    }
}