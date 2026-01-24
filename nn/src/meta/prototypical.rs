//! Prototypical Networks.
//!
//! This module implements Prototypical Networks, a metric-based few-shot learning
//! approach that learns to compute class prototypes and classify by nearest neighbor
//! in the learned embedding space.
//!
//! # Overview
//!
//! Prototypical Networks learn a metric space where classification can be performed
//! by computing class prototypes (mean embeddings) and finding the nearest prototype
//! for query examples. This approach is particularly effective for few-shot learning
//! scenarios where only limited examples per class are available.
//!
//! # Key Components
//!
//! - [`PrototypicalNetwork`]: Main network implementation with encoder and distance metric
//! - [`Episode`]: Few-shot learning task definition with support and query sets
//! - [`FewShotEpisodeGenerator`]: Utility for generating random few-shot episodes
//! - [`DistanceMetric`]: Supported distance/similarity metrics (Euclidean, Cosine, Learned)
//!
//! # Example Usage
//!
//! ```rust
//! use nn::{
//!     meta::{
//!         prototypical::{PrototypicalNetwork, FewShotEpisodeGenerator, DistanceMetric},
//!         Episode,
//!     },
//!     linear::Linear,
//!     Module,
//! };
//! use backend::CpuBackend;
//! use dtype::float::Float32;
//! use storage::DenseStorage;
//! use tensor::Tensor;
//!
//! // Create encoder network
//! let encoder = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 64).unwrap();
//!
//! // Create prototypical network with Euclidean distance
//! let proto_net = PrototypicalNetwork::new(encoder)
//!     .with_distance_metric(DistanceMetric::Euclidean)
//!     .with_scale(1.0)
//!     .with_temperature(1.0);
//!
//! // Create episode generator for 5-way, 5-shot, 15-query tasks
//! let generator = FewShotEpisodeGenerator::new(
//!     class_examples, // Vec of Vec<Tensor> - examples per class
//!     5,              // n_way
//!     5,              // k_shot
//!     15,             // n_query
//! );
//!
//! // Generate and evaluate an episode
//! let episode = generator.generate_episode().unwrap();
//! let prototypes = proto_net.compute_prototypes(&episode.support_set, episode.num_classes).unwrap();
//!
//! // Classify a query example
//! let query_result = proto_net.classify(&episode.query_set[0].0, &prototypes).unwrap();
//! let predicted_class = query_result.iter().enumerate()
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .map(|(i, _)| i).unwrap();
//!
//! // Compute episode accuracy
//! let accuracy = proto_net.episode_accuracy(&episode).unwrap();
//! ```

use rand::Rng;

use crate::core::error::{NNError, Result};
use crate::Module;
use backend::{Backend, DataType, Storage};
use dtype::traits::FloatExt;
use storage::{StorageFromVec, StorageToDense};
// use tensor::ops::arithmetic::scalar_div;
use tensor::{ops::arithmetic, Tensor};

/// Few-shot episode (task) definition
#[derive(Debug, Clone)]
pub struct Episode<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Support set: (input, label) pairs for learning class prototypes
    pub support_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Query set: (input, label) pairs for evaluation
    pub query_set: Vec<(Tensor<B, S, T>, usize)>,
    /// Number of classes in this episode
    pub num_classes: usize,
    /// Episode identifier
    pub episode_id: String,
}

/// Prototypical Network implementation
///
/// A Prototypical Network consists of an encoder network that maps inputs to an
/// embedding space and a distance metric for computing similarities between
/// embeddings and class prototypes.
///
/// # Type Parameters
///
/// * `M`: Encoder network type that implements [`Module`]
/// * `B`: Backend type for tensor operations
/// * `S`: Storage type for tensor data
/// * `T`: Data type for tensor elements
///
/// # Examples
///
/// ```rust
/// use nn::meta::prototypical::{PrototypicalNetwork, DistanceMetric};
/// use nn::linear::Linear;
/// use backend::CpuBackend;
/// use dtype::float::Float32;
/// use storage::DenseStorage;
///
/// let encoder = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 64).unwrap();
/// let proto_net = PrototypicalNetwork::new(encoder)
///     .with_distance_metric(DistanceMetric::Cosine)
///     .with_scale(0.5);
/// ```
#[derive(Debug)]
pub struct PrototypicalNetwork<M, B, S, T> {
    /// Embedding network that maps inputs to feature space
    pub encoder: M,
    /// Distance metric for prototype computation
    pub distance_metric: DistanceMetric,
    /// Scaling factor for distance computation
    pub scale: f64,
    /// Training temperature for softmax
    pub temperature: f64,
    /// Phantom data to indicate usage of generic parameters in impl
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Learned metric (requires additional parameters)
    Learned,
}

impl<M, B, S, T> PrototypicalNetwork<M, B, S, T>
where
    M: Clone + Module<B, S, T>,
    B: Backend<Data = T> + Default,
    S: Storage<T>
        + StorageFromVec<T>
        + StorageToDense<T>
        + Clone
        + Send
        + Sync
        + 'static
        + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType
        + FloatExt
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + std::marker::Copy
        + Into<f64>
        + 'static,
{
    /// Create a new Prototypical Network
    pub fn new(encoder: M) -> Self {
        Self {
            encoder,
            distance_metric: DistanceMetric::Euclidean,
            scale: 1.0,
            temperature: 1.0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set scaling factor
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Compute class prototypes from support set
    ///
    /// This method computes the prototype (mean embedding) for each class by:
    /// 1. Encoding all support examples using the encoder network
    /// 2. Grouping embeddings by class label
    /// 3. Computing the mean embedding for each class
    ///
    /// # Arguments
    ///
    /// * `support_set` - Support examples as (input, class_id) pairs
    /// * `num_classes` - Total number of classes in the episode
    ///
    /// # Returns
    ///
    /// A vector of prototype tensors, one per class
    ///
    /// # Errors
    ///
    /// Returns an error if any class has no support examples or if encoding fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nn::meta::prototypical::PrototypicalNetwork;
    /// # use nn::linear::Linear;
    /// # use backend::CpuBackend;
    /// # use dtype::float::Float32;
    /// # use storage::DenseStorage;
    /// # use tensor::Tensor;
    /// # let encoder = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
    /// # let proto_net = PrototypicalNetwork::new(encoder);
    /// // Create support set with 2 classes, 2 examples each
    /// let support_set = vec![
    ///     (Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), 0),
    ///     (Tensor::from_vec(vec![1.1, 2.1], &[2]).unwrap(), 0),
    ///     (Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(), 1),
    ///     (Tensor::from_vec(vec![3.1, 4.1], &[2]).unwrap(), 1),
    /// ];
    ///
    /// let prototypes = proto_net.compute_prototypes(&support_set, 2).unwrap();
    /// assert_eq!(prototypes.len(), 2); // One prototype per class
    /// ```
    pub fn compute_prototypes(
        &self,
        support_set: &[(Tensor<B, S, T>, usize)],
        num_classes: usize,
    ) -> Result<Vec<Tensor<B, S, T>>> {
        // Group examples by class
        let mut class_examples: Vec<Vec<Tensor<B, S, T>>> = vec![Vec::new(); num_classes];

        // Extract features for each support example
        for (input, class_id) in support_set {
            // Use encoder to extract features
            let features = self.encoder.forward(input)?;
            class_examples[*class_id].push(features);
        }

        // Compute prototypes (mean of features per class)
        let mut prototypes = Vec::new();

        for class_features in class_examples {
            if class_features.is_empty() {
                return Err(NNError::InvalidConfiguration {
                    message: "No examples found for a class".to_string(),
                });
            }

            let mut prototype = class_features[0].clone();
            for features in class_features.iter().skip(1) {
                prototype = arithmetic::add(&prototype, features)?;
            }

            // Average the features
            let count: f64 = num_traits::cast::cast(class_features.len() as f64).unwrap();
            let count_t = Tensor::full_like(&prototype, T::from_f64(count as f64).unwrap())?;
            prototype = arithmetic::div(&prototype, &count_t)?;

            // For prototypical networks, we want the prototype to have shape [feature_dim]
            // Since the encoder outputs [batch_size, feature_dim] and we average over batch_size,
            // we need to remove the batch dimension. For DenseStorage, we can extract the data
            // and create a new tensor with the correct shape.
            // This is a limitation of the generic storage interface - in practice, most
            // neural networks use DenseStorage where reshape is available.

            prototypes.push(prototype);
        }

        Ok(prototypes)
    }

    /// Classify a query example using prototypes
    ///
    /// This method classifies a query example by:
    /// 1. Encoding the query using the encoder network
    /// 2. Computing distances/similarities to all class prototypes
    /// 3. Converting distances to probability distribution using softmax
    ///
    /// # Arguments
    ///
    /// * `query` - Query input tensor to classify
    /// * `prototypes` - Vector of prototype tensors, one per class
    ///
    /// # Returns
    ///
    /// A probability distribution over classes as `Vec<f64>`
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails or if no prototypes are provided
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nn::meta::prototypical::PrototypicalNetwork;
    /// # use nn::linear::Linear;
    /// # use backend::CpuBackend;
    /// # use dtype::float::Float32;
    /// # use storage::DenseStorage;
    /// # use tensor::Tensor;
    /// # let encoder = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
    /// # let proto_net = PrototypicalNetwork::new(encoder);
    /// // Assume we have prototypes and a query
    /// # let prototypes = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap()];
    /// # let query = Tensor::from_vec(vec![1.5, 2.5], &[2]).unwrap();
    ///
    /// let probabilities = proto_net.classify(&query, &prototypes).unwrap();
    /// assert_eq!(probabilities.len(), prototypes.len());
    /// // Probabilities sum to approximately 1.0
    /// let sum: f64 = probabilities.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn classify(
        &self,
        query: &Tensor<B, S, T>,
        prototypes: &[Tensor<B, S, T>],
    ) -> Result<Vec<f64>> {
        // Use encoder to extract features
        let query_features = self.encoder.forward(query)?;

        // Compute distances to all prototypes
        let mut distances = Vec::new();
        for prototype in prototypes {
            let distance = self.compute_distance(&query_features, prototype)?;
            distances.push(distance);
        }

        // Convert distances to probabilities using softmax
        self.distances_to_probabilities(&distances)
    }

    /// Compute distance between two feature vectors
    fn compute_distance(&self, x: &Tensor<B, S, T>, y: &Tensor<B, S, T>) -> Result<f64>
    {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                // Simple Euclidean distance calculation
                let x_data = x.as_slice();
                let y_data = y.as_slice();
                let mut sum = 0.0;
                for (a, b) in x_data.iter().zip(y_data.iter()) {
                    let a_f64: f64 = (*a).into();
                    let b_f64: f64 = (*b).into();
                    let diff = a_f64 - b_f64;
                    sum += diff * diff;
                }
                Ok(sum.sqrt())
            }
            DistanceMetric::Cosine => {
                // Simple cosine distance
                let x_data = x.as_slice();
                let y_data = y.as_slice();
                let mut dot_product = 0.0;
                let mut x_norm = 0.0;
                let mut y_norm = 0.0;

                for (a, b) in x_data.iter().zip(y_data.iter()) {
                    let a_f64: f64 = (*a).into();
                    let b_f64: f64 = (*b).into();
                    dot_product += a_f64 * b_f64;
                    x_norm += a_f64 * a_f64;
                    y_norm += b_f64 * b_f64;
                }

                x_norm = x_norm.sqrt();
                y_norm = y_norm.sqrt();
                let cosine_sim = dot_product / (x_norm * y_norm + 1e-8);
                Ok(1.0 - cosine_sim) // Convert similarity to distance
            }
            DistanceMetric::Learned => {
                // Fall back to Euclidean for now
                let x_data = x.as_slice();
                let y_data = y.as_slice();
                let mut sum = 0.0;
                for (a, b) in x_data.iter().zip(y_data.iter()) {
                    let a_f64: f64 = (*a).into();
                    let b_f64: f64 = (*b).into();
                    let diff = a_f64 - b_f64;
                    sum += diff * diff;
                }
                Ok(sum.sqrt())
            }
        }
    }

    /// Convert distances to probability distribution
    fn distances_to_probabilities(&self, distances: &[f64]) -> Result<Vec<f64>> {
        // Convert distances to similarities (negative distance)
        let similarities: Vec<f64> = distances.iter().map(|&d| -d / self.scale).collect();

        // Apply softmax
        let max_sim = similarities
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let exp_sims: Vec<f64> = similarities
            .iter()
            .map(|&s| ((s - max_sim) / self.temperature).exp())
            .collect();

        let sum_exp = exp_sims.iter().sum::<f64>();
        let probabilities: Vec<f64> = exp_sims.iter().map(|&e| e / sum_exp).collect();

        Ok(probabilities)
    }

    /// Compute episode loss (negative log likelihood)
    pub fn episode_loss(&self, episode: &Episode<B, S, T>) -> Result<f64> {
        let prototypes = self.compute_prototypes(&episode.support_set, episode.num_classes)?;

        let mut total_loss = 0.0;

        for (query_input, true_class) in &episode.query_set {
            let probabilities = self.classify(query_input, &prototypes)?;

            // Compute negative log likelihood loss
            let log_prob = (probabilities[*true_class] + 1e-8).ln();
            total_loss -= log_prob;
        }

        Ok(total_loss / episode.query_set.len() as f64)
    }

    /// Compute accuracy on an episode
    pub fn episode_accuracy(&self, episode: &Episode<B, S, T>) -> Result<f64> {
        let prototypes = self.compute_prototypes(&episode.support_set, episode.num_classes)?;

        let mut num_correct = 0;

        for (query_input, true_class) in &episode.query_set {
            let probabilities = self.classify(query_input, &prototypes)?;

            let predicted_class = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if predicted_class == *true_class {
                num_correct += 1;
            }
        }

        Ok(num_correct as f64 / episode.query_set.len() as f64)
    }

    /// Fine-tune the encoder on an episode (optional)
    pub fn adapt_episode(
        &mut self,
        _episode: &Episode<B, S, T>,
        _num_steps: usize,
        _lr: f64,
    ) -> Result<()> {
        // Simplified adaptation - in practice would update encoder parameters
        Ok(())
    }

    /// Extract features for a batch of inputs
    pub fn encode_batch(&self, inputs: &[Tensor<B, S, T>]) -> Result<Vec<Tensor<B, S, T>>> {
        let mut features = Vec::new();

        for input in inputs {
            // Use encoder to extract features
            let feature = self.encoder.forward(input)?;
            features.push(feature);
        }

        Ok(features)
    }
}

/// Episode generator for few-shot classification tasks
///
/// Generates random few-shot learning episodes from a collection of class examples.
/// Each episode consists of:
/// - Support set: Limited examples per class for learning prototypes
/// - Query set: Examples for evaluation
/// - Random class selection (N-way)
/// - Random example sampling (K-shot, N-query)
///
/// # Type Parameters
///
/// * `B`: Backend type for tensor operations
/// * `S`: Storage type for tensor data
/// * `T`: Data type for tensor elements
///
/// # Examples
///
/// ```rust
/// use nn::meta::prototypical::FewShotEpisodeGenerator;
/// use backend::CpuBackend;
/// use dtype::float::Float32;
/// use storage::DenseStorage;
/// use tensor::Tensor;
///
/// // Create example data for 3 classes, 10 examples each
/// let class_examples = vec![
///     (0..10).map(|_| Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()).collect::<Vec<_>>(),
///     (0..10).map(|_| Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap()).collect::<Vec<_>>(),
///     (0..10).map(|_| Tensor::from_vec(vec![5.0, 6.0], &[2]).unwrap()).collect::<Vec<_>>(),
/// ];
///
/// // Create 5-way, 5-shot, 10-query episode generator
/// let generator = FewShotEpisodeGenerator::new(class_examples, 5, 5, 10);
///
/// // Generate a single episode
/// let episode = generator.generate_episode().unwrap();
/// assert_eq!(episode.num_classes, 5);
/// assert_eq!(episode.support_set.len(), 25); // 5 classes × 5 shots
/// assert_eq!(episode.query_set.len(), 50);   // 5 classes × 10 queries
/// ```
#[derive(Debug)]
pub struct FewShotEpisodeGenerator<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Available classes (each class has multiple examples)
    pub class_examples: Vec<Vec<Tensor<B, S, T>>>,
    /// Number of classes per episode (N-way)
    pub n_way: usize,
    /// Number of support examples per class (K-shot)
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
}

impl<B, S, T> FewShotEpisodeGenerator<B, S, T>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + Clone + Into<f64>,
{
    /// Create a new episode generator
    pub fn new(
        class_examples: Vec<Vec<Tensor<B, S, T>>>,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
    ) -> Self {
        Self {
            class_examples,
            n_way,
            k_shot,
            n_query,
        }
    }

    /// Generate a random few-shot episode
    ///
    /// Randomly selects `n_way` classes and samples `k_shot` support examples
    /// plus `n_query` query examples from each selected class.
    ///
    /// # Returns
    ///
    /// A complete episode ready for few-shot learning evaluation
    ///
    /// # Errors
    ///
    /// Returns an error if any selected class doesn't have enough examples
    /// for the requested k_shot + n_query samples
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nn::meta::prototypical::FewShotEpisodeGenerator;
    /// # use backend::CpuBackend;
    /// # use dtype::float::Float32;
    /// # use storage::DenseStorage;
    /// # use tensor::Tensor;
    /// # let class_examples = vec![
    /// #     (0..10).map(|_| Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()).collect::<Vec<_>>(),
    /// #     (0..10).map(|_| Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap()).collect::<Vec<_>>(),
    /// # ];
    /// # let generator = FewShotEpisodeGenerator::new(class_examples, 2, 3, 5);
    ///
    /// let episode = generator.generate_episode().unwrap();
    /// assert_eq!(episode.support_set.len(), 6);  // 2 classes × 3 shots
    /// assert_eq!(episode.query_set.len(), 10);   // 2 classes × 5 queries
    /// ```
    pub fn generate_episode(&self) -> Result<Episode<B, S, T>>
    where
        B: Backend<Data = T> + Default,
        S: Storage<T> + StorageFromVec<T>,
        T: DataType,
    {
        let mut rng = rand::thread_rng();

        // Select N random classes
        let mut selected_classes = Vec::new();
        let mut available_indices: Vec<usize> = (0..self.class_examples.len()).collect();

        for _ in 0..self.n_way {
            let idx = rng.gen_range(0..available_indices.len());
            let class_idx = available_indices.swap_remove(idx);
            selected_classes.push(class_idx);
        }

        let mut support_set = Vec::new();
        let mut query_set = Vec::new();

        // For each selected class, sample K-shot support and N-query query examples
        for (episode_class_id, &global_class_id) in selected_classes.iter().enumerate() {
            let class_examples = &self.class_examples[global_class_id];

            if class_examples.len() < self.k_shot + self.n_query {
                return Err(NNError::InvalidConfiguration {
                    message: format!("Class {} has insufficient examples", global_class_id),
                });
            }

            // Shuffle examples for this class
            let mut example_indices: Vec<usize> = (0..class_examples.len()).collect();
            for i in (1..example_indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                example_indices.swap(i, j);
            }

            // Add support examples
            for &idx in example_indices.iter().take(self.k_shot) {
                support_set.push((class_examples[idx].clone(), episode_class_id));
            }

            // Add query examples
            for &idx in example_indices.iter().skip(self.k_shot).take(self.n_query) {
                query_set.push((class_examples[idx].clone(), episode_class_id));
            }
        }

        Ok(Episode {
            support_set,
            query_set,
            num_classes: self.n_way,
            episode_id: format!("episode_{}", rng.gen::<u64>()),
        })
    }

    /// Generate multiple episodes
    pub fn generate_episodes(&self, num_episodes: usize) -> Result<Vec<Episode<B, S, T>>> {
        let mut episodes: Vec<Episode<B, S, T>> = Vec::new();

        for _ in 0..num_episodes {
            episodes.push(self.generate_episode()?);
        }

        Ok(episodes)
    }
}

/// Meta-training loop for Prototypical Networks
pub fn check_meta_learning<M, B, S, T>(
    network: &mut PrototypicalNetwork<M, B, S, T>,
    episode: Episode<B, S, T>,
    adaptation_steps: usize,
    adaptation_lr: f64,
) -> Result<Vec<f64>>
where
    M: Clone + Module<B, S, T>,
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType
        + FloatExt
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + std::marker::Copy
        + Into<f64>
        + 'static,
{
    // Adapt the network to this episode
    network.adapt_episode(&episode, adaptation_steps, adaptation_lr)?;

    // Compute loss after adaptation
    let loss = network.episode_loss(&episode)?;
    
    Ok(vec![loss])
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use crate::modules::linear::Linear;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use std::sync::Arc;
    use std::thread;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_prototypical_network_thread_safety() {
        // Create encoder network
        let encoder =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        let proto_net = Arc::new(PrototypicalNetwork::new(encoder));

        // Create support set (10 features to match encoder input)
        let support_set = vec![
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.0); 10],
                    &[1, 10],
                )
                .unwrap(),
                0,
            ),
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.1); 10],
                    &[1, 10],
                )
                .unwrap(),
                0,
            ),
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.0); 10],
                    &[1, 10],
                )
                .unwrap(),
                1,
            ),
        ];

        // Test concurrent prototype computation
        let mut handles = vec![];

        for _ in 0..4 {
            let proto_net_clone = Arc::clone(&proto_net);
            let support_set_clone = support_set.clone();

            let handle = thread::spawn(move || {
                let prototypes = proto_net_clone
                    .compute_prototypes(&support_set_clone, 2)
                    .unwrap();
                assert_eq!(prototypes.len(), 2);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_episode_generator_thread_safety() {
        // Create class examples (using 10 features to match encoder)
        let class_examples = vec![
            vec![
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.0); 10],
                    &[10],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.1); 10],
                    &[10],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(0.9); 10],
                    &[10],
                )
                .unwrap(),
            ],
            vec![
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.0); 10],
                    &[10],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.1); 10],
                    &[10],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(2.9); 10],
                    &[10],
                )
                .unwrap(),
            ],
        ];

        let generator = Arc::new(FewShotEpisodeGenerator::<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >::new(class_examples, 2, 2, 1));

        // Test concurrent episode generation
        let mut handles = vec![];

        for _ in 0..4 {
            let generator_clone = Arc::clone(&generator);

            let handle = thread::spawn(move || {
                let episode = generator_clone.generate_episode().unwrap();
                assert_eq!(episode.num_classes, 2);
                assert_eq!(episode.support_set.len(), 4); // 2 classes × 2 shots
                assert_eq!(episode.query_set.len(), 2); // 2 classes × 1 query
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::linear::Linear;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    type Encoder = Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
    type ProtoNet =
        PrototypicalNetwork<Encoder, CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_prototypical_network_creation() {
        let encoder =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        let proto_net: ProtoNet = PrototypicalNetwork::new(encoder);

        assert_eq!(proto_net.scale, 1.0);
        assert_eq!(proto_net.temperature, 1.0);
    }

    #[test]
    fn test_episode_generator() {
        // Create mock class examples
        let class_examples = vec![
            vec![
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.0), Float32::new(2.0)],
                    &[2],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.1), Float32::new(2.1)],
                    &[2],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(0.9), Float32::new(1.9)],
                    &[2],
                )
                .unwrap(),
            ],
            vec![
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.0), Float32::new(4.0)],
                    &[2],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.1), Float32::new(4.1)],
                    &[2],
                )
                .unwrap(),
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(2.9), Float32::new(3.9)],
                    &[2],
                )
                .unwrap(),
            ],
        ];

        let generator = FewShotEpisodeGenerator::<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >::new(class_examples, 2, 2, 1);

        let episode = generator.generate_episode().unwrap();

        assert_eq!(episode.num_classes, 2);
        assert_eq!(episode.support_set.len(), 4); // 2 classes * 2 shots
        assert_eq!(episode.query_set.len(), 2); // 2 classes * 1 query
    }

    #[test]
    fn test_prototype_computation() {
        let encoder =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 3).unwrap();
        let proto_net = PrototypicalNetwork::new(encoder);

        // Create support set for 2 classes
        let support_set = vec![
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.0), Float32::new(2.0)],
                    &[1, 2],
                )
                .unwrap(),
                0,
            ),
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(1.1), Float32::new(2.1)],
                    &[1, 2],
                )
                .unwrap(),
                0,
            ),
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.0), Float32::new(4.0)],
                    &[1, 2],
                )
                .unwrap(),
                1,
            ),
            (
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    vec![Float32::new(3.1), Float32::new(4.1)],
                    &[1, 2],
                )
                .unwrap(),
                1,
            ),
        ];

        let prototypes = proto_net.compute_prototypes(&support_set, 2).unwrap();
        assert_eq!(prototypes.len(), 2);

        // Check that prototypes have the right dimensions
        for prototype in &prototypes {
            // Should match encoder output dimension [1, 3] (batch_size=1, feature_dim=3)
            let dims = prototype.shape().dims();
            assert_eq!(dims, &[1, 3]);
        }
    }

    #[test]
    fn test_classification() {
        use crate::Parameter;
        let mut encoder =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, 3).unwrap();
        // Set encoder weights to identity-like for predictable mapping
        let weight_data = vec![
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
        ];
        let weight_tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
                weight_data,
                &[3, 2],
                CpuBackend::<Float32>::default(),
            )
            .unwrap();
        encoder.weight = Parameter::new(
            weight_tensor.clone().requires_grad_(true),
            "weight".to_string(),
        );
        encoder.weight_t = Some(
            weight_tensor
                .to_dense_generic()
                .unwrap()
                .transpose(1, 0)
                .unwrap(),
        );
        let bias_data = vec![Float32::new(0.0), Float32::new(0.0), Float32::new(0.0)];
        encoder.bias = Parameter::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
                bias_data,
                &[3],
                CpuBackend::<Float32>::default(),
            )
            .unwrap()
            .requires_grad_(true),
            "bias".to_string(),
        );

        let proto_net: ProtoNet = PrototypicalNetwork::new(encoder);

        // Create simple prototypes (manually set for testing)
        let prototypes =
            vec![
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(vec![Float32::new(0.1), Float32::new(0.9), Float32::new(0.0)], &[3], CpuBackend::<Float32>::default()).unwrap(), // Class 0 prototype (close to query)
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(vec![Float32::new(1.0), Float32::new(0.0), Float32::new(0.0)], &[3], CpuBackend::<Float32>::default()).unwrap(), // Class 1 prototype (far from query)
        ];

        // Test query close to class 0 prototype
        let query =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
                vec![Float32::new(0.1), Float32::new(0.9)],
                &[1, 2],
                CpuBackend::<Float32>::default(),
            )
            .unwrap();
        let probabilities = proto_net.classify(&query, &prototypes).unwrap();

        assert_eq!(probabilities.len(), 2);
        assert!(probabilities[0] > probabilities[1]); // Should prefer class 0
    }
}
