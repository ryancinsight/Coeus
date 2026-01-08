//! Calibration infrastructure for quantization

use crate::core::error::{NNError, Result};

use crate::quantization::core::{CalibrationConfig, CalibrationMethod, CalibrationStats};

use backend::Backend;
use dtype::DataType;
use storage::Storage;
use tensor::Tensor;

use serde::{Deserialize, Serialize};

/// Serializable calibration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCalibrationStats {
    /// Minimum value observed (as f64)
    pub min: f64,
    /// Maximum value observed (as f64)
    pub max: f64,
    /// Mean value (as f64)
    pub mean: f64,
    /// Standard deviation (as f64)
    pub std: f64,
    /// Percentiles (as f64 values)
    pub percentiles: Vec<f64>,
    /// Number of samples used to compute these statistics
    pub sample_count: usize,
}

/// Serializable representation of calibration pipeline state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCalibrationPipeline {
    /// Calibration configuration
    pub config: CalibrationConfig,
    /// Maximum number of samples to keep per layer
    pub max_samples_per_layer: usize,
    /// Statistics per layer (layer_name -> list of serializable stats)
    pub layer_stats: std::collections::HashMap<String, Vec<SerializableCalibrationStats>>,
}

/// Automatic calibration pipeline for quantization
///
/// Collects statistics from multiple tensors and applies optimal calibration methods
/// to determine the best quantization parameters.
#[derive(Debug)]
pub struct CalibrationPipeline<T> {
    /// Collected calibration statistics per layer
    pub layer_stats: std::collections::HashMap<String, Vec<CalibrationStats<T>>>,
    /// Calibration configuration
    pub config: CalibrationConfig,
    /// Maximum number of samples to keep per layer
    pub max_samples_per_layer: usize,
}

impl<T> CalibrationPipeline<T>
where
    T: DataType + Clone + PartialOrd + Into<f64> + From<f64>,
    f64: From<T>,
{
    /// Create a new calibration pipeline
    #[must_use]
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            layer_stats: std::collections::HashMap::new(),
            config,
            max_samples_per_layer: 100, // Default limit
        }
    }

    /// Add calibration data for a layer
    pub fn add_calibration_data(
        &mut self,
        layer_name: &str,
        tensor: &Tensor<impl Backend, impl Storage<T>, T>,
    ) -> Result<()> {
        let data = tensor.as_slice();

        // Collect statistics
        let stats = CalibrationStats::collect(data)?;

        // Add to layer statistics
        self.layer_stats
            .entry(layer_name.to_string())
            .or_insert_with(Vec::new)
            .push(stats);

        // Limit the number of samples per layer
        if let Some(samples) = self.layer_stats.get_mut(layer_name) {
            if samples.len() > self.max_samples_per_layer {
                // Keep only the most recent samples
                let start_idx = samples.len() - self.max_samples_per_layer;
                *samples = samples[start_idx..].to_vec();
            }
        }

        Ok(())
    }

    /// Get optimal quantization parameters for a layer
    pub fn get_optimal_params(&self, layer_name: &str, bits: usize) -> Result<(T, T)> {
        let layer_samples =
            self.layer_stats
                .get(layer_name)
                .ok_or_else(|| NNError::InvalidInput {
                    message: format!("No calibration data found for layer: {}", layer_name),
                })?;

        if layer_samples.is_empty() {
            return Err(NNError::InvalidInput {
                message: format!("No calibration samples found for layer: {}", layer_name),
            });
        }

        // Aggregate statistics across all samples for the layer
        let aggregated_stats = self.aggregate_stats(layer_samples)?;

        // Get optimal parameters using the configured method
        let (scale, zero_point) = aggregated_stats.get_optimal_params(self.config.method, bits);

        Ok((scale, zero_point))
    }

    /// Aggregate statistics across multiple calibration samples
    fn aggregate_stats(&self, samples: &[CalibrationStats<T>]) -> Result<CalibrationStats<T>> {
        if samples.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Cannot aggregate empty statistics".to_string(),
            });
        }

        // For now, use the most recent sample
        // In a more sophisticated implementation, we could combine statistics
        Ok(samples[samples.len() - 1].clone())
    }

    /// Reset calibration data for a specific layer
    pub fn reset_layer(&mut self, layer_name: &str) {
        self.layer_stats.remove(layer_name);
    }

    /// Reset all calibration data
    pub fn reset_all(&mut self) {
        self.layer_stats.clear();
    }

    /// Get calibration summary for debugging
    #[must_use]
    pub fn get_summary(&self) -> std::collections::HashMap<String, usize> {
        self.layer_stats
            .iter()
            .map(|(layer, samples)| (layer.clone(), samples.len()))
            .collect()
    }

    /// Convert to serializable representation
    #[must_use]
    pub fn to_serializable(&self) -> SerializableCalibrationPipeline
    where
        T: Into<f64> + Clone,
        f64: From<T>,
    {
        let mut layer_stats = std::collections::HashMap::new();

        for (layer_name, stats_vec) in &self.layer_stats {
            let serializable_stats: Vec<SerializableCalibrationStats> = stats_vec
                .iter()
                .map(|stats| SerializableCalibrationStats {
                    min: f64::from(stats.min.clone()),
                    max: f64::from(stats.max.clone()),
                    mean: f64::from(stats.mean.clone()),
                    std: f64::from(stats.std.clone()),
                    percentiles: stats
                        .percentiles
                        .iter()
                        .map(|&p| f64::from(p.clone()))
                        .collect(),
                    sample_count: 1, // Each CalibrationStats represents one sample batch
                })
                .collect();

            layer_stats.insert(layer_name.clone(), serializable_stats);
        }

        SerializableCalibrationPipeline {
            config: self.config.clone(),
            max_samples_per_layer: self.max_samples_per_layer,
            layer_stats,
        }
    }

    /// Create from serializable representation
    pub fn from_serializable(serializable: SerializableCalibrationPipeline) -> Result<Self>
    where
        T: From<f64>,
    {
        let mut layer_stats = std::collections::HashMap::new();

        for (layer_name, serializable_stats_vec) in serializable.layer_stats {
            let stats_vec: Vec<CalibrationStats<T>> = serializable_stats_vec
                .into_iter()
                .map(|s| CalibrationStats {
                    min: T::from(s.min),
                    max: T::from(s.max),
                    mean: T::from(s.mean),
                    std: T::from(s.std),
                    percentiles: s.percentiles.into_iter().map(T::from).collect(),
                    histogram: None, // Histogram data is not preserved in serialization
                })
                .collect();

            layer_stats.insert(layer_name, stats_vec);
        }

        Ok(Self {
            layer_stats,
            config: serializable.config,
            max_samples_per_layer: serializable.max_samples_per_layer,
        })
    }
}
