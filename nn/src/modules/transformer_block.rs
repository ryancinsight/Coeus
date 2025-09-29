//! Transformer block module combining attention and MLP
//!
//! This module provides the fundamental building block of transformer architectures,
//! combining causal self-attention with a feed-forward network and residual connections.

use crate::{LayerNorm, Module, Result};
use coeus_backend::{Backend, CpuBackend};
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;

/// Transformer block combining attention and MLP
#[derive(Debug, Clone)]
pub struct Block<T: FloatDtype, B: Backend<T> + Clone + Send + Sync = CpuBackend> {
    /// Layer normalization for attention
    pub ln_1: LayerNorm<T, B>,
    /// Causal self-attention layer
    pub attn: crate::modules::causal_self_attention::CausalSelfAttention<T, B>,
    /// Layer normalization for MLP
    pub ln_2: LayerNorm<T, B>,
    /// MLP block
    pub mlp: crate::modules::mlp::MLP<T, B>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Send + Sync + Default> Block<T, B> {
    /// Create a new transformer block
    ///
    /// # Arguments
    /// * `config` - Attention configuration
    pub fn new(config: crate::modules::attention_config::AttentionConfig) -> Result<Self> {
        let ln_1 = LayerNorm::new(vec![config.n_embd])?;
        let attn = crate::modules::causal_self_attention::CausalSelfAttention::new(config.clone())?;
        let ln_2 = LayerNorm::new(vec![config.n_embd])?;
        let mlp = crate::modules::mlp::MLP::new(config.n_embd)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
}

impl<T: FloatDtype, B: Backend<T> + Clone + Send + Sync + Default> Module<T, B> for Block<T, B> {
    fn forward(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // Transformer block: input -> LN -> Attention -> Residual -> LN -> MLP -> Residual

        // First residual connection: attention
        let attn_norm = self.ln_1.forward(input)?;
        let attn_out = self.attn.forward(&attn_norm)?;
        let residual1 = (input + &attn_out)?;

        // Second residual connection: MLP
        let mlp_norm = self.ln_2.forward(&residual1)?;
        let mlp_out = self.mlp.forward(&mlp_norm)?;
        let output = (&\1 + &\2).unwrap();

        Ok(output.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters_mut());
        params.extend(self.attn.parameters_mut());
        params.extend(self.ln_2.parameters_mut());
        params.extend(self.mlp.parameters_mut());
        params
    }
}

impl<T: FloatDtype> fmt::Display for Block<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Block(attn={}, mlp={})", self.attn, self.mlp)
    }
}


