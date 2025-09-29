//! Transformer Decoder modules
//!
//! This module provides the decoder components of transformer architectures,
//! including individual decoder layers and stacked decoder implementations.

use crate::{GELU, LayerNorm, Linear, Module, Result};
use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;

/// Transformer Decoder Layer
///
/// A single layer of the Transformer decoder consisting of masked self-attention,
/// encoder-decoder cross-attention, and a feed-forward network.
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Send + Sync = CpuBackend> {
    /// Masked self-attention mechanism
    pub self_attn: crate::modules::multihead_attention::MultiHeadAttention<T, B>,
    /// Encoder-decoder cross-attention mechanism
    pub multihead_attn: crate::modules::multihead_attention::MultiHeadAttention<T, B>,
    /// First layer normalization (for self-attention)
    pub norm1: LayerNorm<T, B>,
    /// Second layer normalization (for cross-attention)
    pub norm2: LayerNorm<T, B>,
    /// Third layer normalization (for feed-forward)
    pub norm3: LayerNorm<T, B>,
    /// Feed-forward network
    pub linear1: Linear<T, B>,
    /// Feed-forward activation
    pub activation: GELU,
    /// Feed-forward output projection
    pub linear2: Linear<T, B>,
    /// Dropout probability
    pub dropout: f64,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Send + Sync + Default> TransformerDecoderLayer<T, B> {
    /// Create a new TransformerDecoderLayer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feed-forward network
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Result<Self> {
        let self_attn =
            crate::modules::multihead_attention::MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None)?;
        let multihead_attn =
            crate::modules::multihead_attention::MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None)?;
        let norm1 = LayerNorm::new(vec![d_model])?;
        let norm2 = LayerNorm::new(vec![d_model])?;
        let norm3 = LayerNorm::new(vec![d_model])?;
        let linear1 = Linear::new(d_model, dim_feedforward)?;
        let activation = GELU::new();
        let linear2 = Linear::new(dim_feedforward, d_model)?;

        Ok(Self {
            self_attn,
            multihead_attn,
            norm1,
            norm2,
            norm3,
            linear1,
            activation,
            linear2,
            dropout,
        })
    }
}


impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Send + Sync + Default> Module<T, B>
    for TransformerDecoderLayer<T, B>
{
    fn forward(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // For Module trait, assume self-attention only (no cross-attention)
        // TODO: This needs proper implementation for decoder forward pass
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.multihead_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.multihead_attn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.norm3.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> fmt::Display
    for TransformerDecoderLayer<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerDecoderLayer(d_model={}, nhead={}, dim_feedforward={})",
            self.self_attn.embed_dim, self.self_attn.num_heads, self.linear1.out_features
        )
    }
}

/// Transformer Decoder
///
/// Stacks multiple TransformerDecoderLayer instances to form a complete decoder.
#[derive(Debug, Clone)]
pub struct TransformerDecoder<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Stack of decoder layers
    pub layers: Vec<TransformerDecoderLayer<T>>,
    /// Final layer normalization
    pub norm: Option<LayerNorm<T>>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerDecoder<T> {
    /// Create a new TransformerDecoder
    ///
    /// # Arguments
    /// * `decoder_layer` - The decoder layer to stack
    /// * `num_layers` - Number of decoder layers to stack
    /// * `norm` - Optional final layer normalization
    pub fn new(
        decoder_layer: TransformerDecoderLayer<T>,
        num_layers: usize,
        norm: Option<LayerNorm<T>>,
    ) -> Self {
        let layers = vec![decoder_layer; num_layers];
        Self { layers, norm }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerDecoder<T> {
    /// Forward pass with target sequence and optional memory
    ///
    /// # Arguments
    /// * `tgt` - Target sequence
    /// * `memory` - Optional memory from encoder
    /// * `tgt_mask` - Optional target mask
    /// * `memory_mask` - Optional memory mask
    pub fn forward(
        &self,
        tgt: &Tensor<T, CpuBackend>,
        memory: Option<&Tensor<T, CpuBackend>>,
        tgt_mask: Option<&Tensor<T, CpuBackend>>,
        memory_mask: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<Tensor<T, CpuBackend>> {
        let mut output = tgt.clone();

        // Apply each decoder layer
        // TODO: Fix backend-generic tensor types for decoder layers
        for _layer in &self.layers {
            // output = layer.forward(&output, memory, tgt_mask, memory_mask)?;
        }

        // Apply final normalization if specified
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }

        Ok(output)
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerDecoder<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        // For Module trait, assume self-attention only
        self.forward(input, None, None, None)
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
    for TransformerDecoder<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerDecoder(num_layers={}, norm={})",
            self.layers.len(),
            self.norm.is_some()
        )
    }
}


