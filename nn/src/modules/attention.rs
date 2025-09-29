//! Attention mechanisms for transformers (Legacy monolithic module)
//!
//! **DEPRECATED**: This module contains the original monolithic attention implementations.
//! New code should use the modular attention modules:
//! - `attention_config` for configuration structures
//! - `multihead_attention` for multi-head attention layers
//! - `causal_self_attention` for causal self-attention layers
//! - Legacy transformer components (Block, MLP, Transformer*) remain for compatibility
//!
//! ## Mathematical Foundation
//!
//! ### Self-Attention
//!
//! ```math
//! Attention(Q, K, V) = softmax(QK^T / √d_k)V
//! ```
//!
//! ### Multi-Head Attention
//!
//! ```math
//! MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O
//! head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
//! ```
//!
//! ### Causal Mask
//!
//! For causal (autoregressive) attention:
//!
//! ```math
//! mask[i,j] = 1 if j ≤ i else -∞
//! ```
//!
//! ## References
//!
//! - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
//! - [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

// Legacy transformer components preserved for backward compatibility

use crate::Module;
use coeus_backend::CpuBackend;
use coeus_tensor::{FloatDtype, Tensor};
use std::fmt;

// Re-export legacy monolithic implementations for backward compatibility
// These will be removed in a future version

// Re-export the modular implementations for cleaner API
pub use crate::modules::attention_config::AttentionConfig;
pub use crate::modules::multihead_attention::MultiHeadAttention;
pub use crate::modules::causal_self_attention::CausalSelfAttention;
#[derive(Debug, Clone)]
pub struct MLP<T: FloatDtype> {
    /// First linear layer
    pub c_fc: crate::Linear<T>,
    /// GELU activation
    pub gelu: crate::GELU,
    /// Second linear layer
    pub c_proj: crate::Linear<T>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> MLP<T> {
    /// Create a new MLP block
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    pub fn new(n_embd: usize) -> Self {
        // GPT-2 uses 4x expansion for the intermediate layer
        let intermediate_size = 4 * n_embd;

        let c_fc = crate::Linear::new(n_embd, intermediate_size);
        let gelu = crate::GELU::new();
        let c_proj = crate::Linear::new(intermediate_size, n_embd);

        Self { c_fc, gelu, c_proj }
    }
}

impl<T: FloatDtype> Module<T> for MLP<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // MLP: input -> Linear -> GELU -> Linear -> output
        let hidden = self.c_fc.forward(input)?;
        let activated = self.gelu.forward(&hidden)?;
        let output = self.c_proj.forward(&activated)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters_mut());
        params.extend(self.c_proj.parameters_mut());
        params
    }
}

impl<T: FloatDtype> fmt::Display for MLP<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MLP(c_fc={}, c_proj={})",
            self.c_fc.out_features, self.c_proj.out_features
        )
    }
}

/// Transformer block combining attention and MLP
#[derive(Debug, Clone)]
pub struct Block<T: FloatDtype> {
    /// Layer normalization for attention
    pub ln_1: crate::LayerNorm<T>,
    /// Causal self-attention layer
    pub attn: CausalSelfAttention<T>,
    /// Layer normalization for MLP
    pub ln_2: crate::LayerNorm<T>,
    /// MLP block
    pub mlp: MLP<T>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Block<T> {
    /// Create a new transformer block
    ///
    /// # Arguments
    /// * `config` - Attention configuration
    pub fn new(config: AttentionConfig) -> Self {
        let ln_1 = crate::LayerNorm::new(vec![config.n_embd]);
        let attn = CausalSelfAttention::new(config.clone());
        let ln_2 = crate::LayerNorm::new(vec![config.n_embd]);
        let mlp = MLP::new(config.n_embd);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }
}

impl<T: FloatDtype> Module<T> for Block<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Transformer block: input -> LN -> Attention -> Residual -> LN -> MLP -> Residual

        // First residual connection: attention
        let attn_norm = self.ln_1.forward(input)?;
        let attn_out = self.attn.forward(&attn_norm)?;
        let residual1 = (input + &attn_out)?;

        // Second residual connection: MLP
        let mlp_norm = self.ln_2.forward(&residual1)?;
        let mlp_out = self.mlp.forward(&mlp_norm)?;
        let output = (&residual1 + &mlp_out)?;

        Ok(output.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
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

/// Transformer Encoder Layer
///
/// A single layer of the Transformer encoder consisting of multi-head self-attention
/// and a feed-forward network with residual connections and layer normalization.
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Self-attention mechanism
    pub self_attn: MultiHeadAttention<T>,
    /// First layer normalization (for attention)
    pub norm1: crate::LayerNorm<T>,
    /// Second layer normalization (for feed-forward)
    pub norm2: crate::LayerNorm<T>,
    /// Feed-forward network
    pub linear1: crate::Linear<T>,
    /// Feed-forward activation
    pub activation: crate::GELU,
    /// Feed-forward output projection
    pub linear2: crate::Linear<T>,
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
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Self {
        let self_attn =
            MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None);
        let norm1 = crate::LayerNorm::new(vec![d_model]);
        let norm2 = crate::LayerNorm::new(vec![d_model]);
        let linear1 = crate::Linear::new(d_model, dim_feedforward);
        let activation = crate::GELU::new();
        let linear2 = crate::Linear::new(dim_feedforward, d_model);

        Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            activation,
            linear2,
            dropout,
        }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerEncoderLayer<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Multi-head self-attention with residual connection
        let attn_norm = self.norm1.forward(input)?;
        let attn_output = self.self_attn.forward(&attn_norm)?;
        let residual1 = (input + &attn_output)?;

        // Feed-forward network with residual connection
        let ff_norm = self.norm2.forward(&residual1)?;
        let ff_hidden = self.linear1.forward(&ff_norm)?;
        let ff_activated = self.activation.forward(&ff_hidden)?;
        let ff_output = self.linear2.forward(&ff_activated)?;
        let output = (&residual1 + &ff_output)?;

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub norm: Option<crate::LayerNorm<T>>,
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
        norm: Option<crate::LayerNorm<T>>,
    ) -> Self {
        let layers = vec![encoder_layer; num_layers];
        Self { layers, norm }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerEncoder<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerEncoder(num_layers={}, norm={})",
            self.layers.len(),
            self.norm.is_some()
        )
    }
}

/// Transformer Decoder Layer
///
/// A single layer of the Transformer decoder consisting of masked self-attention,
/// encoder-decoder cross-attention, and a feed-forward network.
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Masked self-attention mechanism
    pub self_attn: MultiHeadAttention<T>,
    /// Encoder-decoder cross-attention mechanism
    pub multihead_attn: MultiHeadAttention<T>,
    /// First layer normalization (for self-attention)
    pub norm1: crate::LayerNorm<T>,
    /// Second layer normalization (for cross-attention)
    pub norm2: crate::LayerNorm<T>,
    /// Third layer normalization (for feed-forward)
    pub norm3: crate::LayerNorm<T>,
    /// Feed-forward network
    pub linear1: crate::Linear<T>,
    /// Feed-forward activation
    pub activation: crate::GELU,
    /// Feed-forward output projection
    pub linear2: crate::Linear<T>,
    /// Dropout probability
    pub dropout: f64,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerDecoderLayer<T> {
    /// Create a new TransformerDecoderLayer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feed-forward network
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Self {
        let self_attn =
            MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None);
        let multihead_attn =
            MultiHeadAttention::new(d_model, nhead, dropout, true, false, false, None, None);
        let norm1 = crate::LayerNorm::new(vec![d_model]);
        let norm2 = crate::LayerNorm::new(vec![d_model]);
        let norm3 = crate::LayerNorm::new(vec![d_model]);
        let linear1 = crate::Linear::new(d_model, dim_feedforward);
        let activation = crate::GELU::new();
        let linear2 = crate::Linear::new(dim_feedforward, d_model);

        Self {
            self_attn,
            multihead_attn,
            norm1,
            norm2,
            norm3,
            linear1,
            activation,
            linear2,
            dropout,
        }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> TransformerDecoderLayer<T> {
    /// Forward pass with target sequence and optional memory (encoder output)
    ///
    /// # Arguments
    /// * `tgt` - Target sequence (decoder input)
    /// * `memory` - Optional memory from encoder
    /// * `tgt_mask` - Optional target mask for self-attention
    /// * `memory_mask` - Optional memory mask for cross-attention
    pub fn forward(
        &self,
        tgt: &Tensor<T, CpuBackend>,
        memory: Option<&Tensor<T, CpuBackend>>,
        tgt_mask: Option<&Tensor<T, CpuBackend>>,
        memory_mask: Option<&Tensor<T, CpuBackend>>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        // Masked self-attention with residual connection
        let self_attn_norm = self.norm1.forward(tgt)?;
        let (self_attn_output, _) = self
            .self_attn
            .forward_mha(
                &self_attn_norm,
                &self_attn_norm,
                &self_attn_norm,
                None,
                false,
                tgt_mask,
                false,
            )
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("Decoder self-attention failed: {:?}", e),
            })?;
        let residual1 = (tgt + &self_attn_output)?;

        // Encoder-decoder cross-attention with residual connection
        let cross_attn_norm = self.norm2.forward(&residual1)?;
        if let Some(memory_tensor) = memory {
            let (cross_attn_output, _) = self
                .multihead_attn
                .forward_mha(
                    &cross_attn_norm,
                    memory_tensor,
                    memory_tensor,
                    None,
                    false,
                    memory_mask,
                    false,
                )
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Decoder cross-attention failed: {:?}", e),
                })?;
            let residual2 = (&residual1 + &cross_attn_output)?;

            // Feed-forward network with residual connection
            let ff_norm = self.norm3.forward(&residual2)?;
            let ff_hidden = self.linear1.forward(&ff_norm)?;
            let ff_activated = self.activation.forward(&ff_hidden)?;
            let ff_output = self.linear2.forward(&ff_activated)?;
            let output = (&residual1 + &ff_output)?;

            Ok(output)
        } else {
            // If no memory, skip cross-attention
            let ff_norm = self.norm3.forward(&residual1)?;
            let ff_hidden = self.linear1.forward(&ff_norm)?;
            let ff_activated = self.activation.forward(&ff_hidden)?;
            let ff_output = self.linear2.forward(&ff_activated)?;
            let output = (&residual1 + &ff_output)?;

            Ok(output)
        }
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for TransformerDecoderLayer<T>
{
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // For Module trait, assume self-attention only (no cross-attention)
        self.forward(input, None, None, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
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

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub norm: Option<crate::LayerNorm<T>>,
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
        norm: Option<crate::LayerNorm<T>>,
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
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        let mut output = tgt.clone();

        // Apply each decoder layer
        for layer in &self.layers {
            output = layer.forward(&output, memory, tgt_mask, memory_mask)?;
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
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerDecoder(num_layers={}, norm={})",
            self.layers.len(),
            self.norm.is_some()
        )
    }
}

/// Transformer (Encoder-Decoder Architecture)
///
/// A complete transformer model consisting of an encoder and decoder.
/// This is the standard architecture used in machine translation and other sequence-to-sequence tasks.
#[derive(Debug, Clone)]
pub struct Transformer<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Transformer encoder
    pub encoder: TransformerEncoder<T>,
    /// Transformer decoder
    pub decoder: TransformerDecoder<T>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Transformer<T> {
    /// Create a new Transformer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `nhead` - Number of attention heads
    /// * `num_encoder_layers` - Number of encoder layers
    /// * `num_decoder_layers` - Number of decoder layers
    /// * `dim_feedforward` - Dimension of feed-forward networks
    /// * `dropout` - Dropout probability
    pub fn new(
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
    ) -> Self {
        let encoder_layer = TransformerEncoderLayer::new(d_model, nhead, dim_feedforward, dropout);
        let encoder = TransformerEncoder::new(encoder_layer, num_encoder_layers, None);

        let decoder_layer = TransformerDecoderLayer::new(d_model, nhead, dim_feedforward, dropout);
        let decoder = TransformerDecoder::new(decoder_layer, num_decoder_layers, None);

        Self { encoder, decoder }
    }

    /// Forward pass for sequence-to-sequence tasks
    ///
    /// # Arguments
    /// * `src` - Source sequence
    /// * `tgt` - Target sequence
    /// * `src_mask` - Optional source mask
    /// * `tgt_mask` - Optional target mask (for masked self-attention)
    /// * `memory_mask` - Optional memory mask (for cross-attention)
    pub fn forward(
        &self,
        src: &Tensor<T, CpuBackend>,
        tgt: &Tensor<T, CpuBackend>,
        _src_mask: Option<&Tensor<T, CpuBackend>>,
        tgt_mask: Option<&Tensor<T, CpuBackend>>,
        memory_mask: Option<&Tensor<T, CpuBackend>>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        // Encode source sequence
        let memory = self.encoder.forward(src)?;

        // Decode target sequence using encoded memory
        let output = self
            .decoder
            .forward(tgt, Some(&memory), tgt_mask, memory_mask)?;

        Ok(output)
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T> for Transformer<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // For Module trait, assume encoder-only operation (no decoder)
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters_mut());
        params.extend(self.decoder.parameters_mut());
        params
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> fmt::Display for Transformer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Transformer(d_model={}, nhead={}, num_encoder_layers={}, num_decoder_layers={}, dim_feedforward={})",
            self.encoder.layers[0].self_attn.embed_dim,
            self.encoder.layers[0].self_attn.num_heads,
            self.encoder.layers.len(),
            self.decoder.layers.len(),
            self.encoder.layers[0].linear1.out_features
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Helper to create zero-filled f64 tensors with given shape for tests
    fn zeros_f64(shape: &[usize]) -> Tensor<f64, CpuBackend> {
        let size = shape.iter().product::<usize>();
        Tensor::from_vec(CpuBackend::default(), vec![0.0f64; size], shape.to_vec()).unwrap()
    }

    #[test]
    fn test_causal_attention_creation() {
        let config = AttentionConfig::default();
        let attention: CausalSelfAttention<f64> = CausalSelfAttention::new(config.clone());

        assert_eq!(attention.config.n_head, config.n_head);
        assert_eq!(attention.config.n_embd, config.n_embd);
    }

    #[test]
    fn test_causal_attention_forward() {
        let config = AttentionConfig {
            n_head: 2,
            n_embd: 8,
            block_size: 4,
            dropout: 0.0,
            causal: true,
        };
        let attention: CausalSelfAttention<f64> = CausalSelfAttention::new(config);

        // Input shape: (batch_size=1, seq_len=3, n_embd=8)
        let input = zeros_f64(&[1, 3, 8]);
        let output = attention.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_mlp_creation() {
        let mlp: MLP<f64> = MLP::new(768);

        assert_eq!(mlp.c_fc.in_features, 768);
        assert_eq!(mlp.c_fc.out_features, 3072); // 4x expansion
        assert_eq!(mlp.c_proj.in_features, 3072);
        assert_eq!(mlp.c_proj.out_features, 768);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp: MLP<f64> = MLP::new(8);

        // Input shape: (batch_size=1, seq_len=3, n_embd=8)
        let input = zeros_f64(&[1, 3, 8]);
        let output = mlp.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_multihead_attention_creation() {
        let mha: MultiHeadAttention<f64> = MultiHeadAttention::new(
            64,    // embed_dim
            8,     // num_heads
            0.1,   // dropout
            true,  // bias
            false, // add_bias_kv
            false, // add_zero_attn
            None,  // kdim
            None,  // vdim
        );

        assert_eq!(mha.embed_dim, 64);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.kdim, 64);
        assert_eq!(mha.vdim, 64);
    }

    #[test]
    fn test_multihead_attention_self_attention() {
        let mha: MultiHeadAttention<f64> = MultiHeadAttention::new(
            64,    // embed_dim
            8,     // num_heads
            0.0,   // dropout
            true,  // bias
            false, // add_bias_kv
            false, // add_zero_attn
            None,  // kdim
            None,  // vdim
        );

        // Input: (batch_size=2, seq_len=4, embed_dim=64)
        let input = zeros_f64(&[2, 4, 64]);
        let (output, weights) = mha
            .forward_mha(&input, &input, &input, None, false, None, false)
            .unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
        // Attention weights should be (batch_size, num_heads, seq_len, seq_len)
        assert_eq!(weights.as_ref().unwrap().shape(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_multihead_attention_cross_attention() {
        let mha: MultiHeadAttention<f64> = MultiHeadAttention::new(
            64,       // embed_dim
            8,        // num_heads
            0.0,      // dropout
            true,     // bias
            false,    // add_bias_kv
            false,    // add_zero_attn
            Some(32), // kdim
            Some(32), // vdim
        );

        // Query: (batch_size=2, tgt_len=3, embed_dim=64)
        let query = zeros_f64(&[2, 3, 64]);
        // Key/Value: (batch_size=2, src_len=5, kdim/vdim=32)
        let key = zeros_f64(&[2, 5, 32]);
        let value = zeros_f64(&[2, 5, 32]);

        let (output, weights) = mha
            .forward_mha(&query, &key, &value, None, false, None, false)
            .unwrap();

        // Output should match query dimensions
        assert_eq!(output.shape(), &[2, 3, 64]);
        // Attention weights should be (batch_size, num_heads, tgt_len, src_len)
        assert_eq!(weights.as_ref().unwrap().shape(), &[2, 8, 3, 5]);
    }

    #[test]
    fn test_multihead_attention_module_trait() {
        let mha: MultiHeadAttention<f64> = MultiHeadAttention::new(
            32,    // embed_dim
            4,     // num_heads
            0.0,   // dropout
            true,  // bias
            false, // add_bias_kv
            false, // add_zero_attn
            None,  // kdim
            None,  // vdim
        );

        // Test Module trait implementation (self-attention)
        let input = zeros_f64(&[1, 3, 32]);
        let output = mha.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 3, 32]);

        // Check parameters: each Linear layer has weight and bias (when bias=true)
        let params = mha.parameters();
        assert_eq!(params.len(), 8); // q_proj, k_proj, v_proj, out_proj (weight + bias each)
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let encoder_layer: TransformerEncoderLayer<f64> = TransformerEncoderLayer::new(
            64,  // d_model
            8,   // nhead
            128, // dim_feedforward
            0.1, // dropout
        );

        // Input: (batch_size=2, seq_len=4, d_model=64)
        let input = zeros_f64(&[2, 4, 64]);
        let output = encoder_layer.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let encoder_layer: TransformerEncoderLayer<f64> =
            TransformerEncoderLayer::new(64, 8, 128, 0.1);
        let encoder: TransformerEncoder<f64> = TransformerEncoder::new(encoder_layer, 2, None);

        // Input: (batch_size=2, seq_len=4, d_model=64)
        let input = zeros_f64(&[2, 4, 64]);
        let output = encoder.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
        assert_eq!(encoder.layers.len(), 2);
    }

    #[test]
    fn test_transformer_decoder_layer() {
        let decoder_layer: TransformerDecoderLayer<f64> = TransformerDecoderLayer::new(
            64,  // d_model
            8,   // nhead
            128, // dim_feedforward
            0.1, // dropout
        );

        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = zeros_f64(&[2, 3, 64]);
        // Memory (encoder output): (batch_size=2, src_len=4, d_model=64)
        let memory = zeros_f64(&[2, 4, 64]);

        let output = decoder_layer
            .forward(&tgt, Some(&memory), None, None)
            .unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
    }

    #[test]
    fn test_transformer_decoder() {
        let decoder_layer: TransformerDecoderLayer<f64> =
            TransformerDecoderLayer::new(64, 8, 128, 0.1);
        let decoder: TransformerDecoder<f64> = TransformerDecoder::new(decoder_layer, 2, None);

        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = zeros_f64(&[2, 3, 64]);
        // Memory (encoder output): (batch_size=2, src_len=4, d_model=64)
        let memory = zeros_f64(&[2, 4, 64]);

        let output = decoder.forward(&tgt, Some(&memory), None, None).unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
        assert_eq!(decoder.layers.len(), 2);
    }

    #[test]
    fn test_transformer() {
        let transformer: Transformer<f64> = Transformer::new(
            64,  // d_model
            8,   // nhead
            2,   // num_encoder_layers
            2,   // num_decoder_layers
            128, // dim_feedforward
            0.1, // dropout
        );

        // Source input: (batch_size=2, src_len=4, d_model=64)
        let src = zeros_f64(&[2, 4, 64]);
        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = zeros_f64(&[2, 3, 64]);

        let output = transformer.forward(&src, &tgt, None, None, None).unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
        assert_eq!(transformer.encoder.layers.len(), 2);
        assert_eq!(transformer.decoder.layers.len(), 2);
    }
}



