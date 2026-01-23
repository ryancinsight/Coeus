//! Entropy-based calibration for quantization

use coeus_error::{Error, Result};
use coeus_error::{NNError, StorageError as CoeusStorageError};
use dtype::DataType;
use num_traits::Float;

/// Entropy-based calibration for quantization parameters
///
/// This method minimizes the information loss (entropy) when quantizing
/// by finding the optimal clipping range that preserves the most information.
pub struct EntropyCalibrator<T: DataType> {
    /// Number of histogram bins for entropy calculation
    pub num_bins: usize,
    /// Collected data samples for calibration
    pub samples: Vec<Vec<T>>,
}

impl<T> EntropyCalibrator<T>
where
    T: DataType + Float + Clone + PartialOrd,
{
    /// Create a new entropy-based calibrator
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            samples: Vec::new(),
        }
    }

    /// Add calibration data
    pub fn add_sample(&mut self, data: Vec<T>) {
        self.samples.push(data);
    }

    /// Compute optimal quantization parameters using entropy minimization
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    ///
    /// # Returns
    /// Optimal (scale, zero_point) parameters
    pub fn compute_params(&self, bits: usize) -> Result<(T, T)> {
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

        let min_val = all_data[0];
        let max_val = all_data[all_data.len() - 1];

        // Try different clipping thresholds and find the one with minimum entropy loss
        let mut best_threshold = max_val;
        let mut min_entropy_loss = T::infinity();

        // Test different percentiles as clipping thresholds
        let percentiles = [0.9999, 0.999, 0.99, 0.95, 0.9, 0.85, 0.8];
        
        for &percentile in &percentiles {
            let threshold_idx = ((percentile * (all_data.len() - 1) as f64) as usize).min(all_data.len() - 1);
            let threshold = all_data[threshold_idx];
            
            // Calculate entropy loss for this threshold
            let entropy_loss = self.calculate_entropy_loss(&all_data, min_val, threshold)?;
            
            if entropy_loss < min_entropy_loss {
                min_entropy_loss = entropy_loss;
                best_threshold = threshold;
            }
        }

        // Compute scale and zero_point from optimal range
        let qmax = (1i64 << bits) - 1;
        let range = best_threshold - min_val;
        
        let scale = if range > T::zero() {
            range / T::from(qmax).unwrap()
        } else {
            T::one()
        };

        // Use symmetric quantization (zero_point = 0) for simplicity
        let zero_point = T::zero();

        Ok((scale, zero_point))
    }

    /// Calculate entropy loss for a given clipping threshold
    fn calculate_entropy_loss(&self, data: &[T], min_val: T, max_val: T) -> Result<T> {
        // Create histogram
        let mut histogram = vec![0usize; self.num_bins];
        let range = max_val - min_val;
        
        if range <= T::zero() {
            return Ok(T::zero());
        }

        let bin_width = range / T::from(self.num_bins).unwrap();

        // Fill histogram
        for &value in data {
            let clamped_value = value.min(max_val).max(min_val);
            let bin_idx = if clamped_value == max_val {
                self.num_bins - 1
            } else {
                let normalized = (clamped_value - min_val) / bin_width;
                normalized.to_usize().unwrap_or(0).min(self.num_bins - 1)
            };
            histogram[bin_idx] += 1;
        }

        // Calculate entropy
        let total_count = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &histogram {
            if count > 0 {
                let probability = count as f64 / total_count;
                entropy -= probability * probability.log2();
            }
        }

        // Calculate entropy loss (higher entropy = lower loss)
        let max_entropy = (self.num_bins as f64).log2();
        let entropy_loss = max_entropy - entropy;

        T::from(entropy_loss).ok_or_else(|| Error::Storage(CoeusStorageError::Quantized("Failed to convert entropy loss to target type".to_string())))
    }

    /// Clear all calibration samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Get number of calibration samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }
}