//! GPT-2 model implementation
//!
//! This module provides a complete GPT-2 model implementation based on the
//! transformer architecture for language modeling tasks.
//!
//! ## Architecture
//!
//! ```text
//! Input -> Token Embeddings -> Position Embeddings -> Dropout
//!       -> Transformer Blocks (N times) -> LayerNorm -> Linear -> Softmax
//! ```
//!
//! ## References
//!
//! - [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
//! - [OpenAI GPT-2](https://github.com/openai/gpt-2)

use crate::modules::attention::{AttentionConfig, Block};
use crate::modules::dropout::Dropout;
use crate::{Embedding, LayerNorm, Module, Result};
use coeus_tensor::{FloatDtype, Tensor};
use std::fmt;

/// GPT-2 model configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Attention configuration
    pub attn_config: AttentionConfig,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of transformer blocks
    pub n_layer: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            attn_config: AttentionConfig::default(),
            vocab_size: 50257, // GPT-2 vocabulary size
            n_layer: 12,       // GPT-2 small has 12 layers
            dropout: 0.1,      // Default dropout probability
        }
    }
}

/// Complete GPT-2 model
#[derive(Debug)]
pub struct GPT2<T: FloatDtype> {
    /// Token embeddings
    pub wte: Embedding<T>,
    /// Position embeddings
    pub wpe: Embedding<T>,
    /// Dropout layer
    pub drop: Dropout<T>,
    /// Transformer blocks
    pub h: Vec<Block<T>>,
    /// Final layer normalization
    pub ln_f: LayerNorm<T>,
    /// Language modeling head
    pub lm_head: crate::Linear<T>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> GPT2<T> {
    /// Create a new GPT-2 model
    ///
    /// # Arguments
    /// * `config` - GPT-2 configuration
    pub fn new(config: GPTConfig) -> Self {
        let wte = Embedding::new(config.vocab_size, config.attn_config.n_embd);
        let wpe = Embedding::new(config.attn_config.block_size, config.attn_config.n_embd);

        let mut h = Vec::new();
        for _ in 0..config.n_layer {
            h.push(Block::new(config.attn_config.clone()));
        }

        let ln_f = LayerNorm::new(vec![config.attn_config.n_embd]); // Final layer norm expects (..., n_embd)
        let lm_head = crate::Linear::new(config.attn_config.n_embd, config.vocab_size);
        let drop =
            Dropout::new(T::from_f64(config.dropout).unwrap_or_else(|| T::from_f64(0.1).unwrap()));

        Self {
            wte,
            wpe,
            drop,
            h,
            ln_f,
            lm_head,
        }
    }

    /// Forward pass for language modeling
    ///
    /// # Arguments
    /// * `input` - Input token indices of shape (batch_size, seq_len)
    /// * `targets` - Optional target token indices for loss computation
    ///
    /// # Returns
    /// If targets is None: logits of shape (batch_size, seq_len, vocab_size)
    /// If targets is Some: (logits, loss) tuple
    pub fn forward_lm(&self, input: &Tensor<T>, targets: Option<&Tensor<T>>) -> Result<Tensor<T>> {
        let input_shape = input.shape();

        // Validate input shape - must be 2D (batch_size, seq_len)
        if input_shape.len() != 2 {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![0, 0], // Variable batch and sequence sizes
                actual: input_shape.to_vec(),
            });
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Token embeddings
        let tok_emb = self.wte.forward(input)?;

        // Position embeddings
        let positions: Vec<T> = (0..seq_len).map(|i| T::from(i as f64).unwrap()).collect();
        // Reshape to (1, seq_len) for embedding lookup
        let pos_input = Tensor::from_vec(positions, vec![1, seq_len]);
        let pos_emb = self.wpe.forward(&pos_input)?;

        // Broadcast position embeddings to match batch size
        // pos_emb has shape [1, seq_len, n_embd], we need [batch_size, seq_len, n_embd]
        let pos_emb_data = pos_emb.data();
        let n_embd = pos_emb.shape()[2];
        let mut pos_emb_broadcast = Vec::new();

        for _ in 0..batch_size {
            pos_emb_broadcast.extend_from_slice(pos_emb_data);
        }
        let pos_emb_batch = Tensor::from_vec(pos_emb_broadcast, vec![batch_size, seq_len, n_embd]);

        // Combine token and position embeddings
        let x = (&tok_emb + &pos_emb_batch)?;

        // Apply dropout
        let x = self.drop.forward(&x)?;

        // Forward through transformer blocks
        let mut hidden = x.clone();
        for block in &self.h {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        let hidden_norm = self.ln_f.forward(&hidden)?;

        // Language modeling head
        let logits = self.lm_head.forward(&hidden_norm)?;

        // Compute loss if targets provided
        if let Some(targets) = targets {
            use crate::losses::CrossEntropyLoss;
            let loss_fn = CrossEntropyLoss::new();
            let loss = loss_fn.forward(&logits, targets)?;
            Ok(loss)
        } else {
            Ok(logits)
        }
    }

    /// Generate text autoregressively
    ///
    /// # Arguments
    /// * `input` - Input token indices of shape (batch_size, seq_len)
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more diverse)
    ///
    /// # Returns
    /// Generated token sequence
    pub fn generate(
        &self,
        input: &Tensor<T>,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Result<Tensor<T>> {
        let mut current_input = input.clone();

        for _ in 0..max_new_tokens {
            // Forward pass to get logits
            let logits = self.forward_lm(&current_input, None)?;

            // Get logits for the last token in the sequence
            let batch_size = logits.shape()[0];
            let seq_len = logits.shape()[1];
            let vocab_size = logits.shape()[2];

            // For simplicity, take the last token's logits
            // Shape: (vocab_size,)
            let start_idx = (batch_size - 1) * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            let last_logits = logits.data()[start_idx..start_idx + vocab_size].to_vec();

            // Apply temperature scaling
            let scaled_logits: Vec<T> = last_logits
                .iter()
                .map(|&x| x / T::from(temperature).unwrap())
                .collect();

            // Apply softmax manually (since tensor softmax only supports 2D)
            // Find max for numerical stability
            let mut max_val = T::neg_infinity();
            for &val in &scaled_logits {
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max) and sum
            let mut exp_sum = T::zero();
            let mut exp_vals = Vec::new();
            for &val in &scaled_logits {
                let exp_val = (val - max_val).exp();
                exp_vals.push(exp_val);
                exp_sum = exp_sum + exp_val;
            }

            // Normalize to get probabilities
            let probs: Vec<T> = exp_vals.iter().map(|&exp_val| exp_val / exp_sum).collect();

            // Sample from the distribution (simplified: greedy decoding for now)
            let mut max_prob = T::neg_infinity();
            let mut next_token = 0;

            for (i, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    next_token = i;
                }
            }

            // Append new token to sequence
            // This is simplified - in practice, you'd need to handle batching properly
            let mut new_input_data = current_input.data().to_vec();
            new_input_data.push(T::from(next_token as f64).unwrap());

            let new_seq_len = current_input.shape()[1] + 1;
            current_input = Tensor::from_vec(new_input_data, vec![batch_size, new_seq_len]);
        }

        Ok(current_input)
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T> for GPT2<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.forward_lm(input, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.wte.parameters());
        params.extend(self.wpe.parameters());
        params.extend(self.drop.parameters()); // Dropout has no parameters, but included for completeness
        for block in &self.h {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.wte.parameters_mut());
        params.extend(self.wpe.parameters_mut());
        params.extend(self.drop.parameters_mut()); // Dropout has no parameters, but included for completeness
        for block in &mut self.h {
            params.extend(block.parameters_mut());
        }
        params.extend(self.ln_f.parameters_mut());
        params.extend(self.lm_head.parameters_mut());
        params
    }
}

impl<T: FloatDtype> fmt::Display for GPT2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPT2(vocab_size={}, n_layer={}, n_embd={}, block_size={})",
            self.wte.vocab_size,
            self.h.len(),
            self.wte.embedding_dim,
            self.wpe.vocab_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_creation() {
        let config = GPTConfig::default();
        let model: GPT2<f32> = GPT2::new(config);

        assert_eq!(model.wte.vocab_size, 50257);
        assert_eq!(model.h.len(), 12);
    }

    #[test]
    fn test_gpt2_forward() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 100,
            n_layer: 2,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Input shape: (batch_size=1, seq_len=3)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let output = model.forward_lm(&input, None).unwrap();

        // Output shape: (batch_size=1, seq_len=3, vocab_size=100)
        assert_eq!(output.shape(), &[1, 3, 100]);
    }

    #[test]
    fn test_gpt2_forward_batch() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 50,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Input shape: (batch_size=2, seq_len=2)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let output = model.forward_lm(&input, None).unwrap();

        // Output shape: (batch_size=2, seq_len=2, vocab_size=50)
        assert_eq!(output.shape(), &[2, 2, 50]);
    }

    #[test]
    fn test_gpt2_forward_single_token() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 100,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Input shape: (batch_size=1, seq_len=1)
        let input = Tensor::from_vec(vec![5.0], vec![1, 1]);
        let output = model.forward_lm(&input, None).unwrap();

        // Output shape: (batch_size=1, seq_len=1, vocab_size=100)
        assert_eq!(output.shape(), &[1, 1, 100]);
    }

    #[test]
    fn test_gpt2_forward_max_sequence() {
        let block_size = 4;
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 100,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Input shape: (batch_size=1, seq_len=block_size)
        let input_data: Vec<f32> = (0..block_size).map(|x| x as f32).collect();
        let input = Tensor::from_vec(input_data, vec![1, block_size]);
        let output = model.forward_lm(&input, None).unwrap();

        // Output shape: (batch_size=1, seq_len=block_size, vocab_size=100)
        assert_eq!(output.shape(), &[1, block_size, 100]);
    }

    #[test]
    fn test_gpt2_parameters() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 100,
            n_layer: 2,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Should have parameters from:
        // - Token embeddings (wte)
        // - Position embeddings (wpe)
        // - 2 transformer blocks (each with attn + mlp + layer norms)
        // - Final layer norm (ln_f)
        // - Language modeling head (lm_head)
        let params = model.parameters();
        assert!(!params.is_empty(), "Model should have parameters");

        // Check that we have a reasonable number of parameters with gradients
        let grad_params = params.iter().filter(|p| p.requires_grad()).count();
        assert!(
            grad_params > 0,
            "At least some model parameters should require gradients for training"
        );

        // Most parameters should require gradients (embeddings, linear layers)
        // LayerNorm parameters might not if they use default None values
        assert!(
            grad_params >= params.len() / 2,
            "At least half of parameters should require gradients"
        );
    }

    #[test]
    fn test_gpt2_module_trait() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 50,
            n_layer: 1,
            dropout: 0.0,
        };
        let mut model: GPT2<f32> = GPT2::new(config);

        // Test that Module trait methods work
        assert!(!model.parameters().is_empty());
        assert!(!model.parameters_mut().is_empty());

        // Test forward through Module trait
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 50]);
    }

    #[test]
    fn test_gpt2_generate_basic() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 20,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Input with single token
        let input = Tensor::from_vec(vec![5.0], vec![1, 1]);
        let generated = model.generate(&input, 2, 1.0).unwrap();

        // Should have generated 3 tokens total (1 input + 2 generated)
        assert_eq!(generated.shape(), &[1, 3]);
    }

    #[test]
    fn test_gpt2_invalid_input_shape() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 100,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Invalid input: 1D tensor instead of 2D
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = model.forward_lm(&input, None);

        // Should return an error for invalid input shape
        assert!(result.is_err());
        if let Err(crate::NNError::ShapeMismatch {
            expected: _,
            actual,
        }) = result
        {
            assert_eq!(actual, vec![3]);
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

    #[test]
    fn test_gpt2_out_of_vocab_token() {
        let config = GPTConfig {
            attn_config: AttentionConfig {
                n_head: 2,
                n_embd: 8,
                block_size: 4,
                dropout: 0.0,
                causal: true,
            },
            vocab_size: 10,
            n_layer: 1,
            dropout: 0.0,
        };
        let model: GPT2<f32> = GPT2::new(config);

        // Token index out of vocabulary range
        let input = Tensor::from_vec(vec![15.0], vec![1, 1]);
        let result = model.forward_lm(&input, None);

        // Should return an error for out-of-vocabulary token
        assert!(result.is_err());
        if let Err(crate::NNError::InvalidInput { message }) = result {
            assert!(message.contains("out of vocabulary"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }
}
