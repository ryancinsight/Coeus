//! Transformer Encoder modules
//!
//! This module provides the encoder components of transformer architectures,
//! including individual encoder layers and stacked encoder implementations.

use crate::{GELU, LayerNorm, Linear, Module, Result};
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;

/// Transformer Encoder Layer
///
/// A single layer of the Transformer encoder consisting of multi-head self-attention
/// and a feed-forward network with residual connections and layer normalization.
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Self-attention mechanism
    pub self_attn: crate::modules::multihead_attention::MultiHeadAttention<T>,
    /// First layer normalization (for attention)
    pub norm1: LayerNorm<T>,
    /// Second layer normalization (for feed-forward)
    pub norm2: LayerNorm<T>,
    /// Feed-forward network
    pub linear1: Linear<T>,
    /// Feed-forward activation
    pub activation: GELU,
    /// Feed-forward output projection
    pub linear2: Linear<T>,
    /// Dropout probability
    pub dropout: f64,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerEncoderLayer<T> {
    /// Create a new TransformerEncoderLayer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feed-forward network (default: 2048)
    /// * `dropout` - Dropout probability (default: 0.1)
    /// * `activation` - Activation function (default: GELU)
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Result<Self> {
        let self_attn =
            crate::modules::multihead_attention::MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None)?;
        let norm1 = LayerNorm::new(vec![d_model])?;
        let norm2 = LayerNorm::new(vec![d_model])?;
        let linear1 = Linear::new(d_model, dim_feedforward)?;
        let activation = GELU::new();
        let linear2 = Linear::new(dim_feedforward, d_model)?;

        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            activation,
            linear2,
            dropout,
        })
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerEncoderLayer<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        // Multi-head self-attention with residual connection
        let attn_norm = self.norm1.forward(input)?;
        let attn_output = self.self_attn.forward(&attn_norm)?;
        let residual1 = (input + &attn_output)?;

        // Feed-forward network with residual connection
        let ff_norm = self.norm2.forward(&residual1)?;
        let ff_hidden = self.linear1.forward(&ff_norm)?;
        let ff_activated = self.activation.forward(&ff_hidden)?;
        let ff_output = self.linear2.forward(&ff_activated)?;
        let output = (&\1 + &\2).unwrap();

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> fmt::Display
    for TransformerEncoderLayer<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerEncoderLayer(d_model={}, nhead={}, dim_feedforward={})",
            self.self_attn.embed_dim, self.self_attn.num_heads, self.linear1.out_features
        )
    }
}

/// Transformer Encoder
///
/// Stacks multiple TransformerEncoderLayer instances to form a complete encoder.
#[derive(Debug, Clone)]
pub struct TransformerEncoder<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Stack of encoder layers
    pub layers: Vec<TransformerEncoderLayer<T>>,
    /// Final layer normalization
    pub norm: Option<LayerNorm<T>>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerEncoder<T> {
    /// Create a new TransformerEncoder
    ///
    /// # Arguments
    /// * `encoder_layer` - The encoder layer to stack
    /// * `num_layers` - Number of encoder layers to stack
    /// * `norm` - Optional final layer normalization
    pub fn new(
        encoder_layer: TransformerEncoderLayer<T>,
        num_layers: usize,
        norm: Option<LayerNorm<T>>,
    ) -> Self {
        let layers = vec![encoder_layer; num_layers];
        Self { layers, norm }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerEncoder<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        let mut output = input.clone();

        // Apply each encoder layer
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }

        // Apply final normalization if specified
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(ref mut norm) = self.norm {
            params.extend(norm.parameters_mut());
        }
        params
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> fmt::Display
    for TransformerEncoder<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerEncoder(num_layers={}, norm={})",
            self.layers.len(),
            self.norm.is_some()
        )
    }
}


