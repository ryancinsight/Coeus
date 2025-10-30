//! Architecture evaluation infrastructure.
//!
//! This module provides infrastructure for evaluating neural architectures,
//! including fitness functions, benchmarks, and performance metrics.

use std::collections::HashMap;
use std::time::Duration;

use super::search_space::Architecture;
use crate::error::Result;

/// Architecture evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Accuracy or primary fitness metric
    pub accuracy: f64,
    /// Loss value
    pub loss: f64,
    /// Number of parameters
    pub num_parameters: usize,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Training time per epoch in seconds
    pub training_time_sec: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Architecture evaluator trait
pub trait ArchitectureEvaluator: std::fmt::Debug {
    /// Evaluate an architecture and return fitness metrics
    fn evaluate(&self, architecture: &Architecture) -> Result<EvaluationResult>;

    /// Get the name of this evaluator
    fn name(&self) -> &str;
}

/// Simple fitness-based evaluator (for testing/development)
#[derive(Debug)]
pub struct SimpleEvaluator {
    /// Base accuracy score
    pub base_accuracy: f64,
    /// Parameter penalty coefficient
    pub param_penalty: f64,
    /// Complexity bonus coefficient
    pub complexity_bonus: f64,
}

impl SimpleEvaluator {
    /// Create a new simple evaluator
    pub fn new(base_accuracy: f64, param_penalty: f64, complexity_bonus: f64) -> Self {
        Self {
            base_accuracy,
            param_penalty,
            complexity_bonus,
        }
    }
}

impl Default for SimpleEvaluator {
    fn default() -> Self {
        Self::new(0.8, 0.001, 0.1)
    }
}

impl ArchitectureEvaluator for SimpleEvaluator {
    fn evaluate(&self, architecture: &Architecture) -> Result<EvaluationResult> {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        // Simulate evaluation with some noise
        let num_params = architecture.num_parameters() as f64;
        let num_layers = architecture.layers.len() as f64;

        // Base accuracy with parameter penalty and complexity bonus
        let accuracy_noise = rng.gen_range(-0.1..0.1);
        let accuracy = (self.base_accuracy + self.complexity_bonus * (num_layers / 10.0).ln()
            - self.param_penalty * (num_params / 1000000.0))
            .clamp(0.0, 1.0)
            + accuracy_noise;

        // Simulate other metrics
        let loss = -accuracy.ln() + rng.gen_range(0.0..0.5);
        let latency_ms = 10.0 + num_params / 100000.0 + rng.gen_range(0.0..5.0);
        let memory_mb = num_params * 4.0 / 1000000.0 + rng.gen_range(0.0..50.0);
        let training_time_sec = latency_ms * 100.0 + rng.gen_range(0.0..10.0);

        let mut metrics = HashMap::new();
        metrics.insert("complexity_score".to_string(), num_layers);
        metrics.insert(
            "efficiency_ratio".to_string(),
            accuracy / num_params.max(1.0),
        );

        Ok(EvaluationResult {
            accuracy: accuracy.clamp(0.0, 1.0),
            loss,
            num_parameters: num_params as usize,
            latency_ms,
            memory_mb,
            training_time_sec,
            metrics,
        })
    }

    fn name(&self) -> &str {
        "SimpleEvaluator"
    }
}

/// Benchmark configuration for architecture evaluation
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Dataset name
    pub dataset: String,
    /// Number of epochs for training
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum evaluation time per architecture
    pub max_time_per_eval: Duration,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset: "cifar10".to_string(),
            epochs: 10,
            batch_size: 128,
            learning_rate: 0.001,
            max_time_per_eval: Duration::from_secs(300), // 5 minutes
            early_stopping_patience: 5,
            use_gpu: true,
        }
    }
}

/// Multi-objective fitness calculator
pub struct MultiObjectiveFitness {
    /// Weights for different objectives
    pub weights: HashMap<String, f64>,
}

impl MultiObjectiveFitness {
    /// Create a new multi-objective fitness calculator
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("accuracy".to_string(), 1.0);
        weights.insert("efficiency".to_string(), 0.3);
        weights.insert("latency_penalty".to_string(), -0.2);
        weights.insert("memory_penalty".to_string(), -0.1);

        Self { weights }
    }

    /// Calculate fitness from evaluation result
    pub fn calculate(&self, result: &EvaluationResult) -> f64 {
        let mut fitness = 0.0;

        // Accuracy (primary objective)
        if let Some(&weight) = self.weights.get("accuracy") {
            fitness += weight * result.accuracy;
        }

        // Efficiency (accuracy per parameter)
        if let Some(&weight) = self.weights.get("efficiency") {
            let efficiency = result.accuracy / (result.num_parameters as f64).max(1.0);
            fitness += weight * efficiency;
        }

        // Latency penalty
        if let Some(&weight) = self.weights.get("latency_penalty") {
            fitness += weight * (1.0 / result.latency_ms.max(0.001));
        }

        // Memory penalty
        if let Some(&weight) = self.weights.get("memory_penalty") {
            fitness += weight * (1.0 / result.memory_mb.max(0.001));
        }

        // Additional custom metrics
        for (metric_name, &value) in &result.metrics {
            if let Some(&weight) = self.weights.get(metric_name) {
                fitness += weight * value;
            }
        }

        fitness
    }

    /// Set weight for an objective
    pub fn set_weight(&mut self, objective: String, weight: f64) {
        self.weights.insert(objective, weight);
    }
}

impl Default for MultiObjectiveFitness {
    fn default() -> Self {
        Self::new()
    }
}

/// Architecture evaluation cache for avoiding redundant evaluations
pub struct EvaluationCache {
    cache: HashMap<String, EvaluationResult>,
    max_size: usize,
}

impl EvaluationCache {
    /// Create a new evaluation cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    /// Get cached result for architecture
    pub fn get(&self, architecture_hash: &str) -> Option<&EvaluationResult> {
        self.cache.get(architecture_hash)
    }

    /// Store evaluation result
    pub fn put(&mut self, architecture_hash: String, result: EvaluationResult) {
        if self.cache.len() >= self.max_size {
            // Remove oldest entry (simple FIFO eviction)
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(architecture_hash, result);
    }

    /// Generate hash for architecture
    pub fn hash_architecture(architecture: &Architecture) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        architecture.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl Default for EvaluationCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Parallel architecture evaluator
pub struct ParallelEvaluator<E: ArchitectureEvaluator> {
    /// Base evaluator
    pub evaluator: E,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Evaluation cache
    pub cache: EvaluationCache,
}

impl<E: ArchitectureEvaluator + Clone + Send + 'static> ParallelEvaluator<E> {
    /// Create a new parallel evaluator
    pub fn new(evaluator: E, num_workers: usize) -> Self {
        Self {
            evaluator,
            num_workers,
            cache: EvaluationCache::default(),
        }
    }

    /// Evaluate multiple architectures in parallel
    pub fn evaluate_batch(&self, architectures: &[&Architecture]) -> Result<Vec<EvaluationResult>> {
        use std::sync::mpsc;
        use std::thread;

        if architectures.is_empty() {
            return Ok(Vec::new());
        }

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Divide work among workers
        let chunk_size = (architectures.len() + self.num_workers - 1) / self.num_workers;

        for worker_id in 0..self.num_workers {
            let start_idx = worker_id * chunk_size;
            let end_idx = (start_idx + chunk_size).min(architectures.len());

            if start_idx >= architectures.len() {
                break;
            }

            let tx = tx.clone();
            let evaluator = self.evaluator.clone();
            let architectures_slice = &architectures[start_idx..end_idx];

            // Clone architectures for thread safety
            let architectures_owned: Vec<Architecture> = architectures_slice
                .iter()
                .map(|&arch| arch.clone())
                .collect();

            let handle = thread::spawn(move || {
                let mut results = Vec::new();

                for architecture in architectures_owned {
                    let _arch_hash = EvaluationCache::hash_architecture(&architecture);

                    // For simplicity, skip cache check in parallel evaluation
                    // TODO: Implement thread-safe cache with Arc<Mutex<>>
                    let result = evaluator.evaluate(&architecture).unwrap_or_else(|_| {
                        // Return default result on error
                        EvaluationResult {
                            accuracy: 0.0,
                            loss: f64::INFINITY,
                            num_parameters: architecture.num_parameters(),
                            latency_ms: f64::INFINITY,
                            memory_mb: f64::INFINITY,
                            training_time_sec: f64::INFINITY,
                            metrics: HashMap::new(),
                        }
                    });

                    results.push(result);
                }

                tx.send((worker_id, results)).unwrap();
            });

            handles.push(handle);
        }

        // Collect results
        let mut all_results = vec![Vec::new(); self.num_workers];
        for _ in 0..handles.len() {
            let (worker_id, results) = rx.recv().unwrap();
            all_results[worker_id] = results;
        }

        // Flatten results
        let mut final_results = Vec::new();
        for worker_results in all_results {
            final_results.extend(worker_results);
        }

        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::super::search_space::{ArchitectureType, LayerSpec};
    use super::*;

    #[test]
    fn test_simple_evaluator() {
        let evaluator = SimpleEvaluator::default();
        let mut architecture = Architecture::new(ArchitectureType::CNN);
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });
        architecture.add_layer(LayerSpec::Linear { out_features: 10 });

        let result = evaluator.evaluate(&architecture).unwrap();

        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert!(result.loss >= 0.0);
        assert!(result.num_parameters > 0);
        assert!(result.latency_ms >= 0.0);
        assert!(result.memory_mb >= 0.0);
    }

    #[test]
    fn test_multi_objective_fitness() {
        let fitness_calc = MultiObjectiveFitness::new();

        let result = EvaluationResult {
            accuracy: 0.85,
            loss: 0.5,
            num_parameters: 1000000,
            latency_ms: 50.0,
            memory_mb: 100.0,
            training_time_sec: 100.0,
            metrics: HashMap::new(),
        };

        let fitness = fitness_calc.calculate(&result);
        assert!(fitness > 0.0); // Should be positive with default weights
    }

    #[test]
    fn test_evaluation_cache() {
        let mut cache = EvaluationCache::new(10);

        let mut architecture = Architecture::new(ArchitectureType::CNN);
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        let arch_hash = EvaluationCache::hash_architecture(&architecture);
        let result = EvaluationResult {
            accuracy: 0.8,
            loss: 0.4,
            num_parameters: 10000,
            latency_ms: 10.0,
            memory_mb: 50.0,
            training_time_sec: 60.0,
            metrics: HashMap::new(),
        };

        cache.put(arch_hash.clone(), result.clone());
        let cached = cache.get(&arch_hash).unwrap();

        assert_eq!(cached.accuracy, result.accuracy);
        assert_eq!(cached.num_parameters, result.num_parameters);
    }
}
