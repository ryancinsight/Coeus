//! NAS utilities and helper functions.
//!
//! This module provides utility functions and helpers for neural architecture search,
//! including search space analysis, architecture comparison, and benchmarking tools.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::search_space::{Architecture, ArchitectureSpace};
use crate::error::{NNError, Result};

/// Architecture comparison and analysis utilities
pub struct ArchitectureAnalyzer;

impl ArchitectureAnalyzer {
    /// Compare two architectures and compute similarity metrics
    pub fn compare_architectures(
        arch1: &Architecture,
        arch2: &Architecture,
    ) -> ArchitectureSimilarity {
        let mut similarity = ArchitectureSimilarity {
            type_match: arch1.architecture_type == arch2.architecture_type,
            ..Default::default()
        };

        // Compare layer counts
        similarity.layer_count_diff =
            (arch1.layers.len() as isize - arch2.layers.len() as isize).unsigned_abs();

        // Compare parameter counts
        let params1 = arch1.num_parameters() as f64;
        let params2 = arch2.num_parameters() as f64;
        similarity.parameter_ratio = params1.max(params2) / params1.min(params2).max(1.0);

        // Compare layer types (simplified)
        let mut layer_type_matches = 0;
        let max_layers = arch1.layers.len().max(arch2.layers.len());

        for i in 0..max_layers {
            if i < arch1.layers.len()
                && i < arch2.layers.len()
                && std::mem::discriminant(&arch1.layers[i])
                    == std::mem::discriminant(&arch2.layers[i])
            {
                layer_type_matches += 1;
            }
        }

        similarity.layer_type_similarity = layer_type_matches as f64 / max_layers as f64;

        // Overall similarity score (weighted combination)
        similarity.overall_similarity = 0.4 * (similarity.type_match as i32 as f64)
            + 0.2 * (1.0 / (1.0 + similarity.layer_count_diff as f64))
            + 0.2 * (1.0 / similarity.parameter_ratio)
            + 0.2 * similarity.layer_type_similarity;

        similarity
    }

    /// Analyze architecture diversity in a population
    pub fn analyze_diversity(population: &[Architecture]) -> PopulationDiversity {
        if population.is_empty() {
            return PopulationDiversity::default();
        }

        let mut similarities = Vec::new();
        let mut unique_architectures = std::collections::HashSet::new();

        // Calculate pairwise similarities
        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                let similarity = Self::compare_architectures(&population[i], &population[j]);
                similarities.push(similarity.overall_similarity);
            }

            // Count unique architectures (simplified by parameter count)
            unique_architectures.insert(population[i].num_parameters());
        }

        let avg_similarity = if similarities.is_empty() {
            0.0
        } else {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        };

        PopulationDiversity {
            average_similarity: avg_similarity,
            uniqueness_ratio: unique_architectures.len() as f64 / population.len() as f64,
            similarity_std: Self::calculate_std(&similarities),
        }
    }

    /// Calculate standard deviation
    fn calculate_std(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }

    /// Find Pareto-optimal architectures (multi-objective optimization)
    pub fn find_pareto_optimal(
        architectures: &[Architecture],
        objectives: &[Box<dyn Fn(&Architecture) -> f64>],
    ) -> Vec<usize> {
        #![allow(clippy::type_complexity)]
        let mut pareto_front = Vec::new();

        for (i, arch) in architectures.iter().enumerate() {
            let mut is_dominated = false;
            let current_objectives: Vec<f64> = objectives.iter().map(|obj| obj(arch)).collect();

            // Check if this architecture is dominated by any other
            for (j, other_arch) in architectures.iter().enumerate() {
                if i == j {
                    continue;
                }

                let other_objectives: Vec<f64> =
                    objectives.iter().map(|obj| obj(other_arch)).collect();

                // Check if other_arch dominates current arch
                let dominates = other_objectives
                    .iter()
                    .zip(&current_objectives)
                    .all(|(other, current)| *other >= *current)
                    && other_objectives
                        .iter()
                        .zip(&current_objectives)
                        .any(|(other, current)| *other > *current);

                if dominates {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                pareto_front.push(i);
            }
        }

        pareto_front
    }
}

/// Architecture similarity metrics
#[derive(Debug, Clone)]
pub struct ArchitectureSimilarity {
    /// Whether architecture types match
    pub type_match: bool,
    /// Difference in layer counts
    pub layer_count_diff: usize,
    /// Ratio of parameter counts (larger/smaller)
    pub parameter_ratio: f64,
    /// Similarity in layer types (0.0 to 1.0)
    pub layer_type_similarity: f64,
    /// Overall similarity score (0.0 to 1.0)
    pub overall_similarity: f64,
}

impl Default for ArchitectureSimilarity {
    fn default() -> Self {
        Self {
            type_match: false,
            layer_count_diff: 0,
            parameter_ratio: 1.0,
            layer_type_similarity: 0.0,
            overall_similarity: 0.0,
        }
    }
}

/// Population diversity metrics
#[derive(Debug, Clone)]
pub struct PopulationDiversity {
    /// Average similarity between architectures
    pub average_similarity: f64,
    /// Ratio of unique architectures
    pub uniqueness_ratio: f64,
    /// Standard deviation of similarities
    pub similarity_std: f64,
}

impl Default for PopulationDiversity {
    fn default() -> Self {
        Self {
            average_similarity: 0.0,
            uniqueness_ratio: 0.0,
            similarity_std: 0.0,
        }
    }
}

/// Search space analysis utilities
pub struct SearchSpaceAnalyzer;

impl SearchSpaceAnalyzer {
    /// Estimate the size of the search space
    pub fn estimate_space_size(search_space: &ArchitectureSpace) -> u128 {
        let mut total_size = 1u128;

        // For each layer position
        for _ in 0..search_space.max_layers {
            let mut layer_combinations = 0u128;

            // Sum over all layer types
            for layer_type in &search_space.layer_types {
                if let Some(range) = search_space.parameter_ranges.get(layer_type) {
                    // Calculate parameter combinations for this layer type
                    let param_combinations = Self::calculate_parameter_combinations(range);
                    layer_combinations = layer_combinations.saturating_add(param_combinations);
                }
            }

            // Multiply by whether layer is present or not (simplified)
            // Use checked multiplication to prevent overflow
            match total_size.checked_mul(layer_combinations.saturating_add(1)) {
                Some(result) => total_size = result,
                None => {
                    // Overflow occurred, return maximum representable value
                    return u128::MAX;
                }
            }
        }

        total_size
    }

    /// Calculate number of parameter combinations for a layer type
    fn calculate_parameter_combinations(range: &super::search_space::ParameterRange) -> u128 {
        let out_channels_combinations = (range.out_channels.1 - range.out_channels.0 + 1) as u128;
        let kernel_size_combinations = (range.kernel_size.1 - range.kernel_size.0 + 1) as u128;
        let stride_combinations = (range.stride.1 - range.stride.0 + 1) as u128;
        let padding_combinations = (range.padding.1 - range.padding.0 + 1) as u128;
        let out_features_combinations = (range.out_features.1 - range.out_features.0 + 1) as u128;
        let num_heads_combinations = (range.num_heads.1 - range.num_heads.0 + 1) as u128;

        // For simplicity, return the maximum (could be more sophisticated)
        out_channels_combinations
            .max(kernel_size_combinations)
            .max(stride_combinations)
            .max(padding_combinations)
            .max(out_features_combinations)
            .max(num_heads_combinations)
    }

    /// Analyze search space coverage
    pub fn analyze_coverage(
        search_space: &ArchitectureSpace,
        architectures: &[Architecture],
    ) -> SearchSpaceCoverage {
        let total_space_size = Self::estimate_space_size(search_space);
        let sampled_architectures = architectures.len() as u128;

        let coverage_ratio = if total_space_size > 0 {
            sampled_architectures as f64 / total_space_size as f64
        } else {
            0.0
        };

        SearchSpaceCoverage {
            total_space_size,
            sampled_architectures,
            coverage_ratio,
            unique_parameter_counts: Self::count_unique_parameter_counts(architectures),
        }
    }

    /// Count unique parameter counts in architectures
    fn count_unique_parameter_counts(architectures: &[Architecture]) -> usize {
        let mut param_counts = std::collections::HashSet::new();
        for arch in architectures {
            param_counts.insert(arch.num_parameters());
        }
        param_counts.len()
    }
}

/// Search space coverage analysis
#[derive(Debug, Clone)]
pub struct SearchSpaceCoverage {
    /// Total estimated size of search space
    pub total_space_size: u128,
    /// Number of sampled architectures
    pub sampled_architectures: u128,
    /// Coverage ratio (sampled / total)
    pub coverage_ratio: f64,
    /// Number of unique parameter counts
    pub unique_parameter_counts: usize,
}

/// NAS benchmarking utilities
pub struct NASBenchmarker {
    /// Benchmark configurations
    pub configs: Vec<BenchmarkConfig>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Benchmark name
    pub name: String,
    /// Dataset size (number of samples)
    pub dataset_size: usize,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Number of classes
    pub num_classes: usize,
    /// Compute budget (max time per evaluation)
    pub max_time: Duration,
}

impl NASBenchmarker {
    /// Create a new benchmarker with standard configurations
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for NASBenchmarker {
    fn default() -> Self {
        let configs = vec![
            BenchmarkConfig {
                name: "CIFAR-10".to_string(),
                dataset_size: 50000,
                input_dims: vec![3, 32, 32],
                num_classes: 10,
                max_time: Duration::from_secs(300),
            },
            BenchmarkConfig {
                name: "ImageNet-tiny".to_string(),
                dataset_size: 100000,
                input_dims: vec![3, 64, 64],
                num_classes: 200,
                max_time: Duration::from_secs(600),
            },
        ];

        Self { configs }
    }
}

impl NASBenchmarker {
    /// Run benchmark on an architecture
    pub fn benchmark_architecture(
        &self,
        architecture: &Architecture,
        config_name: &str,
    ) -> Result<BenchmarkResult> {
        let config = self
            .configs
            .iter()
            .find(|c| c.name == config_name)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Benchmark config '{}' not found", config_name),
            })?;

        // Simulate benchmarking (in real implementation, this would train/evaluate)
        let start_time = Instant::now();

        // Simulate training time based on architecture complexity
        let complexity_factor = (architecture.num_parameters() as f64 / 1000000.0).max(0.1);
        let simulated_training_time = (complexity_factor * 60.0) as u64; // seconds
        std::thread::sleep(Duration::from_millis(simulated_training_time * 10)); // Faster simulation

        let actual_time = start_time.elapsed();

        // Simulate accuracy based on architecture complexity and noise
        let base_accuracy = 0.5 + (1.0 / (1.0 + complexity_factor)) * 0.4;
        let accuracy_noise = (rand::random::<f64>() - 0.5) * 0.1;
        let accuracy = (base_accuracy + accuracy_noise).clamp(0.0, 1.0);

        Ok(BenchmarkResult {
            config_name: config.name.clone(),
            accuracy,
            training_time: actual_time,
            inference_time: Duration::from_millis((complexity_factor * 10.0) as u64),
            memory_usage: architecture.num_parameters() * 4, // Rough estimate in bytes
            metrics: HashMap::new(),
        })
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark configuration name
    pub config_name: String,
    /// Final accuracy achieved
    pub accuracy: f64,
    /// Total training time
    pub training_time: Duration,
    /// Inference time per sample
    pub inference_time: Duration,
    /// Peak memory usage in bytes
    pub memory_usage: usize,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::super::search_space::{ArchitectureType, LayerSpec, LayerType, ParameterRange};
    use super::*;

    #[test]
    fn test_architecture_comparison() {
        let mut arch1 = Architecture::new(ArchitectureType::CNN);
        arch1.add_layer(LayerSpec::Conv2D {
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        let mut arch2 = Architecture::new(ArchitectureType::CNN);
        arch2.add_layer(LayerSpec::Conv2D {
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        let similarity = ArchitectureAnalyzer::compare_architectures(&arch1, &arch2);
        assert!(similarity.overall_similarity > 0.0);
        assert!(similarity.overall_similarity <= 1.0);
    }

    #[test]
    fn test_population_diversity() {
        let mut architectures = Vec::new();

        for i in 0..5 {
            let mut arch = Architecture::new(ArchitectureType::CNN);
            arch.add_layer(LayerSpec::Conv2D {
                out_channels: 32 + i as usize * 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            });
            architectures.push(arch);
        }

        let diversity = ArchitectureAnalyzer::analyze_diversity(&architectures);
        assert!(diversity.average_similarity >= 0.0);
        assert!(diversity.uniqueness_ratio > 0.0);
    }

    #[test]
    fn test_search_space_analysis() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());

        let space_size = SearchSpaceAnalyzer::estimate_space_size(&search_space);
        assert!(space_size > 0);

        let architectures = vec![Architecture::new(ArchitectureType::CNN)];
        let coverage = SearchSpaceAnalyzer::analyze_coverage(&search_space, &architectures);
        assert!(coverage.coverage_ratio >= 0.0);
    }

    #[test]
    fn test_nas_benchmarking() {
        let benchmarker = NASBenchmarker::new();
        let mut architecture = Architecture::new(ArchitectureType::CNN);
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        let result = benchmarker
            .benchmark_architecture(&architecture, "CIFAR-10")
            .unwrap();
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert!(result.training_time > Duration::from_secs(0));
    }
}
