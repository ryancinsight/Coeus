//! Embedding layers for neural networks
//!
//! This module provides embedding layers for converting discrete tokens
//! into continuous vector representations.
//!
//! ## Mathematical Foundation
//!
//! An embedding layer performs a lookup operation:
//!
//! ```math
//! output[i,j] = embedding_matrix[input[i], j]
//! ```
//!
//! Where:
//! - `input` is a tensor of token indices of shape `(batch_size, seq_len)`
//! - `embedding_matrix` is the learnable embedding matrix of shape `(vocab_size, embedding_dim)`
//! - `output` is the embedded tensor of shape `(batch_size, seq_len, embedding_dim)`
//!
//! ## References
//!
//! - [Word2Vec: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
//! - [PyTorch Embedding Layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

use crate::Module;
use coeus_backend::CpuBackend;
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor};
use rand::prelude::*;
use std::fmt;

/// Mode for EmbeddingBag aggregation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EmbeddingBagMode {
    /// Sum embeddings (default)
    #[default]
    Sum,
    /// Average embeddings
    Mean,
    /// Maximum embeddings
    Max,
}

/// Embedding layer for token representations
#[derive(Debug, Clone)]
pub struct Embedding<T: FloatDtype> {
    /// Embedding matrix of shape (vocab_size, embedding_dim)
    pub weight: Tensor<T, CpuBackend>,
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: usize,
    /// Embedding dimension (size of each token vector)
    pub embedding_dim: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Embedding<T> {
    /// Create a new embedding layer
    ///
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens in the vocabulary
    /// * `embedding_dim` - Dimension of each embedding vector
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Embedding;
    ///
    /// let embedding: Embedding<f32> = Embedding::new(50000, 768);
    /// assert_eq!(embedding.vocab_size, 50000);
    /// assert_eq!(embedding.embedding_dim, 768);
    /// ```
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize embeddings with uniform distribution
        // Following common practice: [-sqrt(3/vocab_size), sqrt(3/vocab_size)]
        let bound = (3.0 / vocab_size as f64).sqrt();
        let mut rng = rand::thread_rng();

        let weight_data: Vec<T> = (0..vocab_size * embedding_dim)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let mut weight = Tensor::from_vec(CpuBackend::default(), weight_data, vec![vocab_size, embedding_dim]).unwrap();
        weight.set_requires_grad(true); // Embeddings need gradients for training

        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Create an embedding layer from a pre-trained weight matrix
    ///
    /// # Arguments
    /// * `weight` - Pre-trained embedding matrix of shape (vocab_size, embedding_dim)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Embedding;
    /// use coeus_tensor::Tensor;
    ///
    /// let weight = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
    /// let embedding = Embedding::from_weight(weight);
    /// ```
    pub fn from_weight(weight: Tensor<T, CpuBackend>) -> Self {
        let vocab_size = weight.shape()[0];
        let embedding_dim = weight.shape()[1];

        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }
}

impl<T: FloatDtype> Module<T> for Embedding<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Input should be token indices of shape (batch_size, seq_len)
        // Output will be embeddings of shape (batch_size, seq_len, embedding_dim)

        if input.shape().len() != 2 {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![0, 0], // We don't know exact batch/seq sizes
                actual: input.shape().to_vec(),
            });
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Convert input indices to usize for indexing
        let input_data = input.data();
        let mut output_data = Vec::with_capacity(batch_size * seq_len * self.embedding_dim);

        #[allow(clippy::needless_range_loop)]
        for i in 0..input_data.len() {
            let token_idx = Dtype::to_f64(&input_data[i]).unwrap_or(0.0) as usize;

            if token_idx >= self.vocab_size {
                return Err(crate::NNError::InvalidInput {
                    message: format!(
                        "Token index {} out of vocabulary size {}",
                        token_idx, self.vocab_size
                    ),
                });
            }

            // Copy the embedding vector for this token
            for j in 0..self.embedding_dim {
                let weight_idx = token_idx * self.embedding_dim + j;
                output_data.push(self.weight.data()[weight_idx]);
            }
        }

        let output_shape = vec![batch_size, seq_len, self.embedding_dim];
        let mut output = Tensor::from_vec(CpuBackend::default(), output_data, output_shape).unwrap();

        // Propagate requires_grad flag
        if self.weight.requires_grad() || input.requires_grad() {
            output.set_requires_grad(true);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![&mut self.weight]
    }
}

impl<T: FloatDtype> fmt::Display for Embedding<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding({}, {})", self.vocab_size, self.embedding_dim)
    }
}

/// EmbeddingBag layer for efficient bag-of-words operations
///
/// Computes sums, means, or maxima of embeddings corresponding to bags of indices.
/// This is more memory efficient than using Embedding + aggregation operations.
///
/// ## Mathematical Foundation
///
/// For a bag of indices `[i1, i2, ..., in]` with optional weights `[w1, w2, ..., wn]`:
///
/// ### Sum Mode
/// ```math
/// output[j] = Σ_{k=1 to n} w_k * embedding_matrix[indices[k], j]
/// ```
///
/// ### Mean Mode
/// ```math
/// output[j] = (1/Σ w_k) * Σ_{k=1 to n} w_k * embedding_matrix[indices[k], j]
/// ```
///
/// ### Max Mode
/// ```math
/// output[j] = max_{k=1 to n} embedding_matrix[indices[k], j]
/// ```
///
/// ## References
///
/// - [PyTorch EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)
#[derive(Debug, Clone)]
pub struct EmbeddingBag<T: FloatDtype> {
    /// Embedding matrix of shape (vocab_size, embedding_dim)
    pub weight: Tensor<T, CpuBackend>,
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: usize,
    /// Embedding dimension (size of each token vector)
    pub embedding_dim: usize,
    /// Aggregation mode
    pub mode: EmbeddingBagMode,
    /// Maximum norm for weight normalization (optional)
    pub max_norm: Option<f64>,
    /// p-norm for max_norm computation
    pub norm_type: f64,
    /// Scaling factor for max_norm
    pub scale_grad_by_freq: bool,
    /// Whether to use sparse gradients
    pub sparse: bool,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> EmbeddingBag<T> {
    /// Create a new EmbeddingBag layer
    ///
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens in the vocabulary
    /// * `embedding_dim` - Dimension of each embedding vector
    /// * `mode` - Aggregation mode (Sum, Mean, or Max)
    /// * `max_norm` - Optional maximum norm for weight normalization
    /// * `norm_type` - p-norm for max_norm computation (default: 2.0)
    /// * `scale_grad_by_freq` - Whether to scale gradients by frequency (default: false)
    /// * `sparse` - Whether to use sparse gradients (default: false)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::{EmbeddingBag, EmbeddingBagMode};
    ///
    /// let embedding_bag = EmbeddingBag::<f32>::new(1000, 64, EmbeddingBagMode::Sum, None, 2.0, false, false);
    /// ```
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
        max_norm: Option<f64>,
        norm_type: f64,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> Self {
        // Initialize embeddings with uniform distribution
        // Following common practice: [-sqrt(3/vocab_size), sqrt(3/vocab_size)]
        let bound = (3.0 / vocab_size as f64).sqrt();
        let mut rng = rand::thread_rng();

        let weight_data: Vec<T> = (0..vocab_size * embedding_dim)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let mut weight = Tensor::from_vec(CpuBackend::default(), weight_data, vec![vocab_size, embedding_dim]).unwrap();
        weight.set_requires_grad(true); // Embeddings need gradients for training

        Self {
            weight,
            vocab_size,
            embedding_dim,
            mode,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        }
    }

    /// Create an EmbeddingBag layer from a pre-trained weight matrix
    ///
    /// # Arguments
    /// * `weight` - Pre-trained embedding matrix of shape (vocab_size, embedding_dim)
    /// * `mode` - Aggregation mode
    pub fn from_weight(weight: Tensor<T, CpuBackend>, mode: EmbeddingBagMode) -> Self {
        let vocab_size = weight.shape()[0];
        let embedding_dim = weight.shape()[1];

        Self {
            weight,
            vocab_size,
            embedding_dim,
            mode,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        }
    }
}

impl<T: FloatDtype> Module<T> for EmbeddingBag<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Input should be token indices of shape (batch_size, seq_len) or flattened (total_indices,)
        // For EmbeddingBag, we typically expect flattened indices with offsets
        // But for simplicity, we'll handle (batch_size, seq_len) format

        if input.shape().len() == 1 {
            // Flattened indices: (total_indices,)
            self.forward_bag(input, None, None)
        } else if input.shape().len() == 2 {
            // Batched indices: (batch_size, seq_len)
            // Convert to flattened format for bag processing
            let batch_size = input.shape()[0];
            let seq_len = input.shape()[1];

            // Create offsets for each batch (assuming equal sequence lengths)
            let offsets = (0..batch_size)
                .map(|i| T::from((i * seq_len) as f64).unwrap())
                .collect::<Vec<T>>();
            let offsets_tensor = Tensor::from_vec(CpuBackend::default(), offsets, vec![batch_size]).unwrap();

            // Flatten input
            let input_flat = input.reshape(vec![batch_size * seq_len])?;

            self.forward_bag(&input_flat, Some(&offsets_tensor), None)
        } else {
            Err(crate::NNError::ShapeMismatch {
                expected: vec![0], // Flexible for 1D or 2D
                actual: input.shape().to_vec(),
            })
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![&mut self.weight]
    }
}

impl<T: FloatDtype> EmbeddingBag<T> {
    /// Forward pass with explicit bag structure
    ///
    /// # Arguments
    /// * `input` - Flattened tensor of token indices of shape (total_indices,)
    /// * `offsets` - Optional offsets tensor indicating start of each bag of shape (batch_size,)
    /// * `per_sample_weights` - Optional per-sample weights of shape (total_indices,)
    ///
    /// # Returns
    /// Tensor of shape (batch_size, embedding_dim)
    pub fn forward_bag(
        &self,
        input: &Tensor<T, CpuBackend>,
        offsets: Option<&Tensor<T, CpuBackend>>,
        per_sample_weights: Option<&Tensor<T, CpuBackend>>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input.shape().len() != 1 {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![0], // 1D tensor
                actual: input.shape().to_vec(),
            });
        }

        let total_indices = input.shape()[0];

        // Determine batch size and bag sizes
        let (batch_size, _bag_sizes) = if let Some(offsets_tensor) = offsets {
            if offsets_tensor.shape().len() != 1 {
                return Err(crate::NNError::ShapeMismatch {
                    expected: vec![0], // 1D offsets
                    actual: offsets_tensor.shape().to_vec(),
                });
            }

            let batch_size = offsets_tensor.shape()[0];
            let offsets_data = offsets_tensor.data();

            // Calculate bag sizes from offsets
            let mut bag_sizes = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let start = if i == 0 {
                    0
                } else {
                    Dtype::to_f64(&offsets_data[i - 1]).unwrap_or(0.0) as usize
                };
                let end = if i == batch_size - 1 {
                    total_indices
                } else {
                    Dtype::to_f64(&offsets_data[i]).unwrap_or(0.0) as usize
                };
                bag_sizes.push(end - start);
            }

            (batch_size, bag_sizes)
        } else {
            // Assume single bag containing all indices
            (1, vec![total_indices])
        };

        // Validate per-sample weights if provided
        if let Some(weights) = per_sample_weights {
            if weights.shape() != input.shape() {
                return Err(crate::NNError::ShapeMismatch {
                    expected: input.shape().to_vec(),
                    actual: weights.shape().to_vec(),
                });
            }
        }

        // Initialize output data
        let mut output_data = vec![T::zero(); batch_size * self.embedding_dim];

        // Process each bag
        let input_data = input.data();
        let weight_data = self.weight.data();

        for bag_idx in 0..batch_size {
            let bag_start = if let Some(offsets_tensor) = offsets {
                let offsets_data = offsets_tensor.data();
                if bag_idx == 0 {
                    0
                } else {
                    Dtype::to_f64(&offsets_data[bag_idx - 1]).unwrap_or(0.0) as usize
                }
            } else {
                0
            };

            let bag_end = if let Some(offsets_tensor) = offsets {
                let offsets_data = offsets_tensor.data();
                if bag_idx == batch_size - 1 {
                    total_indices
                } else {
                    Dtype::to_f64(&offsets_data[bag_idx]).unwrap_or(0.0) as usize
                }
            } else {
                total_indices
            };

            // For mean mode, calculate weight sum
            let weight_sum = if self.mode == EmbeddingBagMode::Mean {
                if let Some(weights) = per_sample_weights {
                    (bag_start..bag_end)
                        .map(|i| Dtype::to_f64(&weights.data()[i]).unwrap_or(1.0))
                        .sum::<f64>()
                } else {
                    (bag_end - bag_start) as f64
                }
            } else {
                1.0 // Not used for sum/max modes
            };

            // Aggregate embeddings for this bag
            for (idx_pos, &token_val) in input_data
                .iter()
                .enumerate()
                .skip(bag_start)
                .take(bag_end - bag_start)
            {
                let token_idx = Dtype::to_f64(&token_val).unwrap_or(0.0) as usize;

                if token_idx >= self.vocab_size {
                    return Err(crate::NNError::InvalidInput {
                        message: format!(
                            "Token index {} out of vocabulary size {}",
                            token_idx, self.vocab_size
                        ),
                    });
                }

                // Get weight for this sample
                let sample_weight = if let Some(weights) = per_sample_weights {
                    Dtype::to_f64(&weights.data()[idx_pos]).unwrap_or(1.0)
                } else {
                    1.0
                };

                // Aggregate embedding
                for emb_dim in 0..self.embedding_dim {
                    let weight_idx = token_idx * self.embedding_dim + emb_dim;
                    let embedding_val = weight_data[weight_idx];
                    let weighted_val =
                        <T as Dtype>::from_f64(Dtype::to_f64(&embedding_val).unwrap_or(0.0) * sample_weight)
                            .unwrap();

                    let output_idx = bag_idx * self.embedding_dim + emb_dim;

                    match self.mode {
                        EmbeddingBagMode::Sum => {
                            // Add to accumulator
                            let current_val = output_data[output_idx];
                            output_data[output_idx] = <T as Dtype>::from_f64(
                                Dtype::to_f64(&current_val).unwrap_or(0.0)
                                    + Dtype::to_f64(&weighted_val).unwrap_or(0.0),
                            )
                            .unwrap();
                        }
                        EmbeddingBagMode::Mean => {
                            // Add to accumulator (will be normalized later)
                            let current_val = output_data[output_idx];
                            output_data[output_idx] = <T as Dtype>::from_f64(
                                Dtype::to_f64(&current_val).unwrap_or(0.0)
                                    + Dtype::to_f64(&weighted_val).unwrap_or(0.0),
                            )
                            .unwrap();
                        }
                        EmbeddingBagMode::Max => {
                            // Take maximum
                            let current_val = output_data[output_idx];
                            if Dtype::to_f64(&weighted_val).unwrap_or(f64::NEG_INFINITY)
                                > Dtype::to_f64(&current_val).unwrap_or(f64::NEG_INFINITY)
                            {
                                output_data[output_idx] = weighted_val;
                            }
                        }
                    }
                }
            }

            // For mean mode, normalize by weight sum
            if self.mode == EmbeddingBagMode::Mean && weight_sum > 0.0 {
                for emb_dim in 0..self.embedding_dim {
                    let output_idx = bag_idx * self.embedding_dim + emb_dim;
                    let current_val = output_data[output_idx];
                    output_data[output_idx] =
                        <T as Dtype>::from_f64(Dtype::to_f64(&current_val).unwrap_or(0.0) / weight_sum)
                            .unwrap();
                }
            }
        }

        let mut output = Tensor::from_vec(CpuBackend::default(), output_data, vec![batch_size, self.embedding_dim]).unwrap();

        // Propagate requires_grad flag
        if self.weight.requires_grad() {
            output.set_requires_grad(true);
        }

        Ok(output)
    }
}

impl<T: FloatDtype> fmt::Display for EmbeddingBag<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EmbeddingBag({}, {}, mode={:?})",
            self.vocab_size, self.embedding_dim, self.mode
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let embedding: Embedding<f32> = Embedding::new(100, 64);

        assert_eq!(embedding.vocab_size, 100);
        assert_eq!(embedding.embedding_dim, 64);
        assert_eq!(embedding.weight.shape(), &[100, 64]);
    }

    #[test]
    fn test_embedding_forward() {
        let embedding: Embedding<f32> = Embedding::new(10, 5);

        // Create input with token indices [0, 1, 2]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let output = embedding.forward(&input).unwrap();

        // Output should be shape (3, 5)
        assert_eq!(output.shape(), &[3, 5]);
    }

    #[test]
    fn test_embedding_parameters() {
        let embedding: Embedding<f32> = Embedding::new(50, 128);

        assert_eq!(embedding.parameters().len(), 1);
        assert_eq!(embedding.parameters()[0].shape(), &[50, 128]);

        let mut embedding_mut: Embedding<f64> = Embedding::new(50, 128);
        assert_eq!(embedding_mut.parameters_mut().len(), 1);
    }

    #[test]
    fn test_embedding_invalid_token() {
        let embedding: Embedding<f32> = Embedding::new(10, 5);

        // Token index out of vocabulary
        let input = Tensor::from_vec(CpuBackend::default(), vec![15.0], vec![1]).unwrap();
        let result = embedding.forward(&input);

        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_bag_creation() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(100, 64, EmbeddingBagMode::Sum, None, 2.0, false, false);

        assert_eq!(embedding_bag.vocab_size, 100);
        assert_eq!(embedding_bag.embedding_dim, 64);
        assert_eq!(embedding_bag.mode, EmbeddingBagMode::Sum);
        assert_eq!(embedding_bag.weight.shape(), &[100, 64]);
    }

    #[test]
    fn test_embedding_bag_sum_mode() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(10, 3, EmbeddingBagMode::Sum, None, 2.0, false, false);

        // Input: bag with indices [0, 1, 2]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let output = embedding_bag.forward_bag(&input, None, None).unwrap();

        // Output should be shape (1, 3) - single bag
        assert_eq!(output.shape(), &[1, 3]);

        // For sum mode, output should be sum of embeddings 0, 1, and 2
        let expected_sum: Vec<f32> = (0..3)
            .map(|dim| {
                embedding_bag.weight.data()[dim]
                    + embedding_bag.weight.data()[3 + dim]
                    + embedding_bag.weight.data()[6 + dim]
            })
            .collect();

        for (i, &expected) in expected_sum.iter().enumerate() {
            assert!((output.data()[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_embedding_bag_mean_mode() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(10, 3, EmbeddingBagMode::Mean, None, 2.0, false, false);

        // Input: bag with indices [0, 1]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0], vec![2]).unwrap();
        let output = embedding_bag.forward_bag(&input, None, None).unwrap();

        // Output should be shape (1, 3) - single bag
        assert_eq!(output.shape(), &[1, 3]);

        // For mean mode, output should be average of embeddings 0 and 1
        let expected_mean: Vec<f32> = (0..3)
            .map(|dim| {
                (embedding_bag.weight.data()[dim] + embedding_bag.weight.data()[3 + dim]) / 2.0
            })
            .collect();

        for (i, &expected) in expected_mean.iter().enumerate() {
            assert!((output.data()[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_embedding_bag_max_mode() {
        // Create a simple embedding matrix with known values
        let vocab_size = 3;
        let embed_dim = 3;
        let weight_data = vec![
            1.0, 2.0, 3.0, // index 0
            4.0, 1.0, 6.0, // index 1
            2.0, 5.0, 1.0, // index 2
        ];
        let weight = Tensor::from_vec(CpuBackend::default(), weight_data, vec![vocab_size, embed_dim]).unwrap();

        let embedding_bag = EmbeddingBag::from_weight(weight, EmbeddingBagMode::Max);

        // Input: bag with indices [0, 1, 2]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let output = embedding_bag.forward_bag(&input, None, None).unwrap();

        // Output should be shape (1, 3) - single bag
        assert_eq!(output.shape(), &[1, 3]);

        // For max mode, output should be element-wise max
        // dim 0: max(1.0, 4.0, 2.0) = 4.0
        // dim 1: max(2.0, 1.0, 5.0) = 5.0
        // dim 2: max(3.0, 6.0, 1.0) = 6.0
        assert!((output.data()[0] - 4.0f32).abs() < 1e-6);
        assert!((output.data()[1] - 5.0f32).abs() < 1e-6);
        assert!((output.data()[2] - 6.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_bag_with_offsets() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(10, 2, EmbeddingBagMode::Sum, None, 2.0, false, false);

        // Two bags: [0, 1] and [2]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let offsets = Tensor::from_vec(CpuBackend::default(), vec![0.0, 2.0], vec![2]).unwrap(); // offsets at 0 and 2

        let output = embedding_bag
            .forward_bag(&input, Some(&offsets), None)
            .unwrap();

        // Output should be shape (2, 2) - two bags
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_embedding_bag_with_weights() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(10, 2, EmbeddingBagMode::Sum, None, 2.0, false, false);

        // Input: bag with indices [0, 1]
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.0, 1.0], vec![2]).unwrap();
        // Weights: [2.0, 3.0]
        let weights = Tensor::from_vec(CpuBackend::default(), vec![2.0, 3.0], vec![2]).unwrap();

        let output = embedding_bag
            .forward_bag(&input, None, Some(&weights))
            .unwrap();

        // Output should be shape (1, 2) - single bag
        assert_eq!(output.shape(), &[1, 2]);

        // Check weighted sum
        for dim in 0..2 {
            let expected =
                embedding_bag.weight.data()[dim] * 2.0 + embedding_bag.weight.data()[2 + dim] * 3.0;
            assert!((output.data()[dim] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_embedding_bag_parameters() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(50, 128, EmbeddingBagMode::Mean, None, 2.0, false, false);

        assert_eq!(embedding_bag.parameters().len(), 1);
        assert_eq!(embedding_bag.parameters()[0].shape(), &[50, 128]);
    }

    #[test]
    fn test_embedding_bag_invalid_token() {
        let embedding_bag: EmbeddingBag<f32> =
            EmbeddingBag::new(10, 3, EmbeddingBagMode::Sum, None, 2.0, false, false);

        // Token index out of vocabulary
        let input = Tensor::from_vec(CpuBackend::default(), vec![15.0], vec![1]).unwrap();
        let result = embedding_bag.forward_bag(&input, None, None);

        assert!(result.is_err());
    }
}



