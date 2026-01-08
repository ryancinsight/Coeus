//! Meta-Learning Benchmarks.
//!
//! This module provides comprehensive benchmarking tools for meta-learning algorithms,
//! including standard few-shot learning datasets, evaluation metrics, and statistical analysis.
//! This supports the unified research framework for systematic algorithm comparison.

use crate::core::error::{NNError, Result};
use rand::Rng;

#[cfg(feature = "research")]
pub use super::adapters::{
    MAMLAdapter, MAMLAgentFactory, PrototypicalAdapter, PrototypicalAgentFactory,
};

/// Few-shot learning dataset
#[derive(Debug)]
pub struct FewShotDataset {
    /// Dataset name
    pub name: String,
    /// Number of classes
    pub num_classes: usize,
    /// Number of examples per class
    pub examples_per_class: usize,
    /// Feature dimension
    pub feature_dim: usize,
    /// Training classes
    pub train_classes: Vec<usize>,
    /// Validation classes
    pub val_classes: Vec<usize>,
    /// Test classes
    pub test_classes: Vec<usize>,
    /// Class examples (class_id -> examples)
    pub class_examples: Vec<Vec<Vec<f64>>>,
}

impl FewShotDataset {
    /// Create a synthetic few-shot dataset
    pub fn synthetic(
        name: &str,
        num_classes: usize,
        examples_per_class: usize,
        feature_dim: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Split classes into train/val/test (70%/15%/15%)
        let train_split = (num_classes as f64 * 0.7) as usize;
        let val_split = (num_classes as f64 * 0.85) as usize;

        let train_classes: Vec<usize> = (0..train_split).collect();
        let val_classes: Vec<usize> = (train_split..val_split).collect();
        let test_classes: Vec<usize> = (val_split..num_classes).collect();

        // Generate synthetic examples for each class
        let mut class_examples = Vec::new();

        for _class_id in 0..num_classes {
            let mut class_data = Vec::new();

            // Generate class prototype (center)
            let prototype: Vec<f64> = (0..feature_dim)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .collect();

            // Generate examples around prototype
            for _ in 0..examples_per_class {
                let mut example = Vec::new();
                for &proto_val in &prototype {
                    // Add Gaussian noise
                    let noise = rng.gen::<f64>() * 0.1;
                    example.push(proto_val + noise);
                }
                class_data.push(example);
            }

            class_examples.push(class_data);
        }

        Self {
            name: name.to_string(),
            num_classes,
            examples_per_class,
            feature_dim,
            train_classes,
            val_classes,
            test_classes,
            class_examples,
        }
    }

    /// Sample a few-shot episode
    pub fn sample_episode(
        &self,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        split: DatasetSplit,
    ) -> Result<Episode> {
        let available_classes = match split {
            DatasetSplit::Train => &self.train_classes,
            DatasetSplit::Validation => &self.val_classes,
            DatasetSplit::Test => &self.test_classes,
        };

        if available_classes.len() < n_way {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "Not enough classes in {} split: {} available, {} needed",
                    split.name(),
                    available_classes.len(),
                    n_way
                ),
            });
        }

        // Sample N classes
        let mut selected_classes = Vec::new();
        let mut available_indices: Vec<usize> = (0..available_classes.len()).collect();

        for _ in 0..n_way {
            let idx = rand::random::<usize>() % available_indices.len();
            let class_idx = available_indices.swap_remove(idx);
            selected_classes.push(available_classes[class_idx]);
        }

        // Sample K-shot + N-query examples per class
        let mut support_set = Vec::new();
        let mut query_set = Vec::new();

        for (episode_class_id, &global_class_id) in selected_classes.iter().enumerate() {
            let class_examples = &self.class_examples[global_class_id];

            if class_examples.len() < k_shot + n_query {
                return Err(NNError::InvalidConfiguration {
                    message: format!(
                        "Class {} has insufficient examples: {} available, {} needed",
                        global_class_id,
                        class_examples.len(),
                        k_shot + n_query
                    ),
                });
            }

            // Shuffle examples
            let mut example_indices: Vec<usize> = (0..class_examples.len()).collect();
            for i in (1..example_indices.len()).rev() {
                let j = rand::random::<usize>() % (i + 1);
                example_indices.swap(i, j);
            }

            // Add support examples
            for &idx in example_indices.iter().take(k_shot) {
                let features = class_examples[idx].clone();
                support_set.push((features, episode_class_id));
            }

            // Add query examples
            for &idx in example_indices.iter().skip(k_shot).take(n_query) {
                let features = class_examples[idx].clone();
                query_set.push((features, episode_class_id));
            }
        }

        Ok(Episode {
            support_set,
            query_set,
            num_classes: n_way,
            episode_id: format!("{}_{}_way_{}_shot", self.name, n_way, k_shot),
        })
    }

    /// Get dataset statistics
    pub fn statistics(&self) -> DatasetStatistics {
        DatasetStatistics {
            name: self.name.clone(),
            num_classes: self.num_classes,
            examples_per_class: self.examples_per_class,
            feature_dim: self.feature_dim,
            train_classes: self.train_classes.len(),
            val_classes: self.val_classes.len(),
            test_classes: self.test_classes.len(),
            total_examples: self.num_classes * self.examples_per_class,
        }
    }
}

#[derive(Debug)]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

impl DatasetSplit {
    pub fn name(&self) -> &'static str {
        match self {
            DatasetSplit::Train => "train",
            DatasetSplit::Validation => "validation",
            DatasetSplit::Test => "test",
        }
    }
}

#[derive(Debug)]
pub struct DatasetStatistics {
    pub name: String,
    pub num_classes: usize,
    pub examples_per_class: usize,
    pub feature_dim: usize,
    pub train_classes: usize,
    pub val_classes: usize,
    pub test_classes: usize,
    pub total_examples: usize,
}

/// Meta-learning benchmark suite
#[derive(Debug)]
pub struct MetaLearningBenchmark {
    /// Available datasets
    pub datasets: Vec<FewShotDataset>,
    /// Benchmark configurations (N-way, K-shot)
    pub configurations: Vec<BenchmarkConfig>,
    /// Benchmark results
    pub results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Dataset name
    pub dataset: String,
    /// N-way classification
    pub n_way: usize,
    /// K-shot learning
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
    /// Number of episodes to evaluate
    pub num_episodes: usize,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark configuration
    pub config: BenchmarkConfig,
    /// Accuracy results (one per episode)
    pub accuracies: Vec<f64>,
    /// Average accuracy
    pub mean_accuracy: f64,
    /// Standard deviation
    pub std_accuracy: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
}

impl MetaLearningBenchmark {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for MetaLearningBenchmark {
    fn default() -> Self {
        let datasets = vec![
            FewShotDataset::synthetic("miniImageNet", 64, 600, 512),
            FewShotDataset::synthetic("tieredImageNet", 351, 1300, 512),
            FewShotDataset::synthetic("CIFAR-FS", 64, 600, 512),
            FewShotDataset::synthetic("FC100", 60, 100, 512),
        ];

        let configurations = vec![
            BenchmarkConfig {
                dataset: "miniImageNet".to_string(),
                n_way: 5,
                k_shot: 1,
                n_query: 15,
                num_episodes: 600,
            },
            BenchmarkConfig {
                dataset: "miniImageNet".to_string(),
                n_way: 5,
                k_shot: 5,
                n_query: 15,
                num_episodes: 600,
            },
            BenchmarkConfig {
                dataset: "tieredImageNet".to_string(),
                n_way: 5,
                k_shot: 1,
                n_query: 15,
                num_episodes: 600,
            },
            BenchmarkConfig {
                dataset: "tieredImageNet".to_string(),
                n_way: 5,
                k_shot: 5,
                n_query: 15,
                num_episodes: 600,
            },
        ];

        Self {
            datasets,
            configurations,
            results: Vec::new(),
        }
    }
}

impl MetaLearningBenchmark {
    /// Run benchmark evaluation
    pub fn run_benchmark<F>(&mut self, mut evaluator: F) -> Result<Vec<BenchmarkResult>>
    where
        F: FnMut(&FewShotDataset, usize, usize, usize, DatasetSplit) -> Result<f64> + Send + Sync,
    {
        let mut results = Vec::new();

        for config in &self.configurations {
            // Find the dataset
            let dataset = self
                .datasets
                .iter()
                .find(|d| d.name == config.dataset)
                .ok_or_else(|| NNError::InvalidConfiguration {
                    message: format!("Dataset '{}' not found", config.dataset),
                })?;

            let mut accuracies = Vec::new();

            // Evaluate on multiple episodes
            for _ in 0..config.num_episodes {
                let accuracy = evaluator(
                    dataset,
                    config.n_way,
                    config.k_shot,
                    config.n_query,
                    DatasetSplit::Test,
                )?;
                accuracies.push(accuracy);
            }

            // Compute statistics
            let mean_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
            let variance = accuracies
                .iter()
                .map(|&x| (x - mean_accuracy).powi(2))
                .sum::<f64>()
                / accuracies.len() as f64;
            let std_accuracy = variance.sqrt();

            // 95% confidence interval (assuming normal distribution)
            let confidence_margin = 1.96 * std_accuracy / (config.num_episodes as f64).sqrt();
            let confidence_interval = (
                mean_accuracy - confidence_margin,
                mean_accuracy + confidence_margin,
            );

            let result = BenchmarkResult {
                config: config.clone(),
                accuracies,
                mean_accuracy,
                std_accuracy,
                confidence_interval,
            };

            results.push(result);
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Get benchmark summary
    pub fn summary(&self) -> BenchmarkSummary {
        let mut total_episodes = 0;
        let mut total_accuracy = 0.0;

        for result in &self.results {
            total_episodes += result.config.num_episodes;
            total_accuracy += result.mean_accuracy * result.config.num_episodes as f64;
        }

        let overall_accuracy = if total_episodes > 0 {
            total_accuracy / total_episodes as f64
        } else {
            0.0
        };

        BenchmarkSummary {
            total_datasets: self.datasets.len(),
            total_configurations: self.configurations.len(),
            total_results: self.results.len(),
            total_episodes,
            overall_accuracy,
        }
    }

    /// Print benchmark results
    pub fn print_results(&self) {
        println!("Meta-Learning Benchmark Results");
        println!("================================");

        for result in &self.results {
            println!("\nDataset: {}", result.config.dataset);
            println!(
                "Configuration: {}-way {}-shot",
                result.config.n_way, result.config.k_shot
            );
            println!("Episodes: {}", result.config.num_episodes);
            println!(
                "Accuracy: {:.2} ± {:.2}%",
                result.mean_accuracy * 100.0,
                result.std_accuracy * 100.0
            );
            println!(
                "95% CI: [{:.2}, {:.2}]%",
                result.confidence_interval.0 * 100.0,
                result.confidence_interval.1 * 100.0
            );
        }

        let summary = self.summary();
        println!("\nSummary:");
        println!("Total datasets: {}", summary.total_datasets);
        println!("Total configurations: {}", summary.total_configurations);
        println!("Total episodes: {}", summary.total_episodes);
        println!("Overall accuracy: {:.2}%", summary.overall_accuracy * 100.0);
    }
}

#[derive(Debug)]
pub struct BenchmarkSummary {
    pub total_datasets: usize,
    pub total_configurations: usize,
    pub total_results: usize,
    pub total_episodes: usize,
    pub overall_accuracy: f64,
}

/// Episode definition for benchmarks
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set: (features, class_id) pairs
    pub support_set: Vec<(Vec<f64>, usize)>,
    /// Query set: (features, class_id) pairs
    pub query_set: Vec<(Vec<f64>, usize)>,
    /// Number of classes in this episode
    pub num_classes: usize,
    /// Episode identifier
    pub episode_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_few_shot_dataset_creation() {
        let dataset = FewShotDataset::synthetic("test", 10, 100, 50);

        assert_eq!(dataset.name, "test");
        assert_eq!(dataset.num_classes, 10);
        assert_eq!(dataset.examples_per_class, 100);
        assert_eq!(dataset.feature_dim, 50);
        assert_eq!(dataset.class_examples.len(), 10);

        for class_examples in &dataset.class_examples {
            assert_eq!(class_examples.len(), 100);
            for example in class_examples {
                assert_eq!(example.len(), 50);
            }
        }
    }

    #[test]
    fn test_episode_sampling() {
        let dataset = FewShotDataset::synthetic("test", 20, 50, 10);

        let episode = dataset
            .sample_episode(5, 5, 10, DatasetSplit::Train)
            .unwrap();

        assert_eq!(episode.num_classes, 5);
        assert_eq!(episode.support_set.len(), 25); // 5 classes * 5 shots
        assert_eq!(episode.query_set.len(), 50); // 5 classes * 10 queries

        // Check that support and query sets have different class IDs within episode
        let support_classes: std::collections::HashSet<_> = episode
            .support_set
            .iter()
            .map(|(_, class)| *class)
            .collect();
        let query_classes: std::collections::HashSet<_> =
            episode.query_set.iter().map(|(_, class)| *class).collect();

        assert_eq!(support_classes.len(), 5);
        assert_eq!(query_classes.len(), 5);
        assert_eq!(support_classes, query_classes);
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = MetaLearningBenchmark::new();

        assert!(!benchmark.datasets.is_empty());
        assert!(!benchmark.configurations.is_empty());

        // Check that standard datasets are included
        let dataset_names: Vec<String> =
            benchmark.datasets.iter().map(|d| d.name.clone()).collect();
        assert!(dataset_names.contains(&"miniImageNet".to_string()));
        assert!(dataset_names.contains(&"tieredImageNet".to_string()));
        assert!(dataset_names.contains(&"CIFAR-FS".to_string()));
        assert!(dataset_names.contains(&"FC100".to_string()));
    }

    #[test]
    fn test_benchmark_evaluation() {
        let mut benchmark = MetaLearningBenchmark::new();

        // Mock evaluator that returns random accuracy
        let evaluator =
            |_: &FewShotDataset, _: usize, _: usize, _: usize, _: DatasetSplit| -> Result<f64> {
                Ok(rand::random::<f64>() * 0.5 + 0.5) // Random accuracy between 0.5 and 1.0
            };

        let results = benchmark.run_benchmark(evaluator).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            assert!(result.mean_accuracy >= 0.0 && result.mean_accuracy <= 1.0);
            assert!(result.std_accuracy >= 0.0);
            assert!(result.confidence_interval.0 <= result.confidence_interval.1);
        }

        let summary = benchmark.summary();
        assert_eq!(summary.total_results, results.len());
        assert!(summary.overall_accuracy >= 0.0 && summary.overall_accuracy <= 1.0);
    }
}
