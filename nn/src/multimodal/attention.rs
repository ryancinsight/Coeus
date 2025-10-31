//! # Cross-Modal Attention Mechanisms
//!
//! Implementation of attention mechanisms that enable cross-modal interaction
//! between different input modalities in multimodal processing.

use std::collections::HashMap;
use crate::error::{NNError, Result};
use crate::attention::MultiHeadAttention;
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use crate::dropout::Dropout;
use crate::functional::linear;
use tensor::Tensor;
use backend::Backend;
use storage::Storage;
use dtype::DataType;
use super::modality::{Modality};

/// Cross-modal attention mechanism with temporal processing
#[derive(Debug)]
pub struct CrossModalAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default,
    T: DataType,
{
    /// Query modality and layer
    pub query_modality: Modality,
    /// Key/Value modalities
    pub kv_modalities: Vec<Modality>,
    /// Multi-head attention mechanism
    pub attention: MultiHeadAttention<B, S, T>,
    /// Query projection for cross-domain adaptation
    pub query_proj: Option<Linear<B, S, T>>,
    /// Key projection
    pub key_proj: Option<Linear<B, S, T>>,
    /// Value projection
    pub value_proj: Option<Linear<B, S, T>>,
    /// Output projection
    pub out_proj: Linear<B, S, T>,
    /// Layer normalization
    pub norm: LayerNorm<B, S, T>,
    /// Dropout layer
    pub dropout: Dropout,
    /// Input dimensions for each modality
    input_dims: HashMap<Modality, usize>,
    /// Hidden dimension
    hidden_dim: usize,
}

impl<B, S, T> CrossModalAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default,
    T: DataType,
{
    /// Create new cross-modal attention
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        query_modality: Modality,
        kv_modalities: Vec<Modality>,
    ) -> Result<Self> {
        let attention = MultiHeadAttention::new(num_heads, hidden_dim)?;
        let out_proj = Linear::new(hidden_dim, hidden_dim)?;
        let norm = LayerNorm::new(hidden_dim, 1e-6)?;
        let dropout = Dropout::new(0.1);

        Ok(Self {
            query_modality,
            kv_modalities,
            attention,
            query_proj: None,
            key_proj: None,
            value_proj: None,
            out_proj,
            norm,
            dropout,
            input_dims: HashMap::new(),
            hidden_dim,
        })
    }

    /// Configure with modality-specific input dimensions
    pub fn with_modality_dims(mut self, input_dims: HashMap<Modality, usize>) -> Result<Self> {
        self.input_dims = input_dims.clone();

        // Create projections for cross-domain adaptation
        // Always create projections for proper cross-modal alignment
        for (modality, &input_dim) in &input_dims {
            if *modality == self.query_modality {
                self.query_proj = Some(Linear::new(input_dim, self.hidden_dim)?);
            }
            if self.kv_modalities.contains(modality) {
                self.key_proj = Some(Linear::new(input_dim, self.hidden_dim)?);
                self.value_proj = Some(Linear::new(input_dim, self.hidden_dim)?);
            }
        }
        Ok(self)
    }

    /// Forward pass for cross-modal attention
    pub fn forward(
        &self,
        query: &Tensor<B, S, T>,
        keys: &HashMap<Modality, Tensor<B, S, T>>,
        values: &HashMap<Modality, Tensor<B, S, T>>,
        mask: Option<&Tensor<B, S, T>>,
    ) -> Result<Tensor<B, S, T>> {
        // Project query if needed
        let query_proj = if let Some(ref proj) = self.query_proj {
            linear(query, &proj.weight, proj.bias.as_ref())?
        } else {
            query.clone()
        };

        // Concatenate all KV modalities
        let mut key_list = Vec::new();
        let mut value_list = Vec::new();

        for modality in &self.kv_modalities {
            if let (Some(key), Some(value)) = (keys.get(modality), values.get(modality)) {
                let key_proj = if let Some(ref proj) = self.key_proj {
                    linear(key, &proj.weight, proj.bias.as_ref())?
                } else {
                    key.clone()
                };

                let value_proj = if let Some(ref proj) = self.value_proj {
                    linear(value, &proj.weight, proj.bias.as_ref())?
                } else {
                    value.clone()
                };

                key_list.push(key_proj);
                value_list.push(value_proj);
            }
        }

        if key_list.is_empty() {
            return Err(NNError::InvalidInput("No valid key-value pairs for cross-modal attention".into()));
        }

        // Concatenate along sequence dimension (assuming batch x seq x hidden)
        // Use tensor concatenation for proper cross-modal attention
        let keys_concat = tensor::ops::tensor_ops::concatenate_tensors(&key_list, 1)?;

        let values_concat = if !value_list.is_empty() {
            // Concatenate value tensors along sequence dimension (dim=1)
            tensor::ops::tensor_ops::concatenate_tensors(&value_list, 1)?
        } else {
            return Err(NNError::InvalidInput("No value tensors to concatenate".into()));
        };

        // Apply multi-head attention
        let attn_output = self.attention.forward(&query_proj, &keys_concat, &values_concat, mask)?;

        // Apply output projection and dropout
        let output = linear(&attn_output, &self.out_proj.weight, self.out_proj.bias.as_ref())?;
        let output = crate::functional::dropout(&output, Some(self.dropout.p), Some(self.dropout.training), Some(false))?;

        // Add residual connection and layer norm
        let residual = query + &output;
        let normalized = self.norm.forward(&residual)?;

        Ok(normalized)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = self.attention.num_parameters() +
                         self.out_proj.num_parameters() +
                         self.norm.num_parameters();

        if let Some(ref proj) = self.query_proj {
            total += proj.num_parameters();
        }
        if let Some(ref proj) = self.key_proj {
            total += proj.num_parameters();
        }
        if let Some(ref proj) = self.value_proj {
            total += proj.num_parameters();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use storage::DenseStorage;
    use dtype::float::Float32;

    #[test]
    fn test_cross_modal_attention_creation() {
        let attention = CrossModalAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            768, 12, Modality::Vision, vec![Modality::Language]
        ).unwrap();

        assert_eq!(attention.query_modality, Modality::Vision);
        assert_eq!(attention.kv_modalities, vec![Modality::Language]);
        assert_eq!(attention.hidden_dim, 768);
    }
}


