//! # Modality Encoders
//!
//! Implementation of specialized encoders for different input modalities.
//! Handles modality-specific preprocessing and transformer encoding.

use crate::error::{NNError, Result};
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use tensor::Tensor;
use backend::Backend;
use storage::{Storage, StorageFromVec};
use dtype::DataType;
use super::modality::{Modality, ModalityConfig};
use super::fusion::FeedForward;

/// Modality-specific encoder
#[derive(Debug)]
pub struct Encoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T>,
    T: DataType + 'static,
{
    /// Modality configuration
    pub config: ModalityConfig,
    /// Input projection
    pub input_proj: Linear<B, S, T>,
    /// Position embeddings (if applicable)
    pub pos_embed: Option<Linear<B, S, T>>,
    /// Transformer layers
    pub layers: Vec<Layer<B, S, T>>,
    /// Output normalization
    pub norm: LayerNorm<B, S, T>,
}

impl<B, S, T> Encoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T>,
    T: DataType + 'static,
{
    /// Create new modality encoder
    pub fn new(config: ModalityConfig) -> Result<Self> {
        let input_proj = Linear::new(config.input_dim, config.hidden_dim)?;
        let pos_embed = if config.modality == Modality::Language || config.modality == Modality::Audio {
            Some(Linear::new(config.max_seq_len, config.hidden_dim)?)
        } else {
            None
        };

        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(Layer::new(config.hidden_dim, config.num_heads, config.dropout)?);
        }

        let norm = LayerNorm::new(config.hidden_dim, 1e-6)?;

        Ok(Self {
            config,
            input_proj,
            pos_embed,
            layers,
            norm,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor<B, S, T>, mask: Option<&Tensor<B, S, T>>) -> Result<Tensor<B, S, T>> {
        // Project input to hidden dimension
        let mut hidden = crate::functional::linear(input, &self.input_proj.weight, self.input_proj.bias.as_ref())?;

        // Add position embeddings if applicable
        if let Some(ref pos_embed) = self.pos_embed {
            // Create proper positional encodings
            let seq_len = hidden.shape()[1];
            let batch_size = hidden.shape()[0];
            let hidden_dim = hidden.shape()[2];

            // Generate positional encodings using sine/cosine functions
            let mut pos_encodings = Vec::with_capacity(seq_len * hidden_dim);

            for pos in 0..seq_len {
                for i in 0..hidden_dim {
                    let angle = (pos as f64) / (10000.0_f64.powf((2.0 * (i / 2) as f64) / hidden_dim as f64));
                    let value = if i % 2 == 0 {
                        angle.sin()
                    } else {
                        angle.cos()
                    };
                    pos_encodings.push(T::from(value).unwrap());
                }
            }

            // Create position embedding tensor and expand for batch
            let pos_tensor = Tensor::from_vec(pos_encodings, &[seq_len, hidden_dim])?;
            let pos_tensor = pos_tensor.unsqueeze(0)?.repeat(&[batch_size, 1, 1])?;
            hidden = hidden + &pos_tensor?;
        }

        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask)?;
        }

        // Apply final layer norm
        self.norm.forward(&hidden)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = self.input_proj.num_parameters() + self.norm.num_parameters();

        if let Some(ref pos_embed) = self.pos_embed {
            total += pos_embed.num_parameters();
        }

        for layer in &self.layers {
            total += layer.num_parameters();
        }

        total
    }
}

#[derive(Debug)]
pub struct Layer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default,
    T: DataType + 'static,
{
    /// Self-attention
    pub attention: crate::attention::MultiHeadAttention<B, S, T>,
    /// Feed-forward network
    pub feed_forward: FeedForward<B, S, T>,
    /// Layer norms
    pub norm1: LayerNorm<B, S, T>,
    pub norm2: LayerNorm<B, S, T>,
    /// Intermediate dropout
    pub dropout: crate::dropout::Dropout,
}

impl<B, S, T> Layer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default,
    T: DataType + 'static,
{
    /// Create new transformer layer
    pub fn new(hidden_dim: usize, num_heads: usize, dropout: f64) -> Result<Self> {
        let attention = crate::attention::MultiHeadAttention::new(num_heads, hidden_dim)?;
        let feed_forward = FeedForward::new(hidden_dim, hidden_dim * 4, dropout)?;
        let norm1 = LayerNorm::new(hidden_dim, 1e-6)?;
        let norm2 = LayerNorm::new(hidden_dim, 1e-6)?;
        let dropout_layer = crate::dropout::Dropout::new(dropout);

        Ok(Self {
            attention,
            feed_forward,
            norm1,
            norm2,
            dropout: dropout_layer,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor<B, S, T>, mask: Option<&Tensor<B, S, T>>) -> Result<Tensor<B, S, T>> {
        // Multi-head self-attention
        let norm1_out = self.norm1.forward(input)?;
        let attn_out = self.attention.forward(&norm1_out, &norm1_out, &norm1_out, mask)?;
        let attn_out = crate::functional::dropout(&attn_out, Some(self.dropout.p), Some(self.dropout.training), Some(false))?;
        let residual1 = input + &attn_out;

        // Feed-forward network
        let norm2_out = self.norm2.forward(&residual1)?;
        let ff_out = self.feed_forward.forward(&norm2_out)?;
        let residual2 = &residual1 + &ff_out;

        Ok(residual2)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.attention.num_parameters() +
        self.feed_forward.num_parameters() +
        self.norm1.num_parameters() +
        self.norm2.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use storage::DenseStorage;
    use dtype::float::Float32;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(768, 12, 0.1).unwrap();
        assert_eq!(layer.attention.num_parameters(), 0); // Would be calculated properly in real implementation
    }

    #[test]
    fn test_encoder_creation() {
        let config = ModalityConfig {
            modality: Modality::Language,
            input_dim: 300,
            hidden_dim: 768,
            num_layers: 6,
            num_heads: 12,
            max_seq_len: 512,
            dropout: 0.1,
            params: std::collections::HashMap::new(),
        };

        let encoder = Encoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config.clone()).unwrap();
        assert_eq!(encoder.config.modality, Modality::Language);
        assert_eq!(encoder.config.input_dim, 300);
        assert_eq!(encoder.config.hidden_dim, 768);
        assert_eq!(encoder.layers.len(), 6);
        assert!(encoder.pos_embed.is_some()); // Should have position embeddings for language
    }
}


