//! Attention mechanisms for transformers
//!
//! This module provides multi-head self-attention and causal self-attention
//! mechanisms used in transformer architectures like GPT-2.
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

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor};
use std::fmt;

/// Configuration for attention layers
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_head: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Maximum sequence length
    pub block_size: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use causal masking
    pub causal: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            n_head: 12,
            n_embd: 768,
            block_size: 1024,
            dropout: 0.1,
            causal: true,
        }
    }
}

/// Multi-Head Attention layer (PyTorch-compatible)
///
/// Implements the standard multi-head attention mechanism used in transformers.
/// Supports both self-attention and cross-attention modes.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T: FloatDtype> {
    /// Query projection matrix
    pub q_proj: crate::Linear<T>,
    /// Key projection matrix
    pub k_proj: crate::Linear<T>,
    /// Value projection matrix
    pub v_proj: crate::Linear<T>,
    /// Output projection matrix
    pub out_proj: crate::Linear<T>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Key/Value embedding dimension (for cross-attention)
    pub kdim: usize,
    /// Value embedding dimension (for cross-attention)
    pub vdim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use bias in projections
    pub bias: bool,
    /// Whether to add bias for causal masking
    pub add_bias_kv: Option<Tensor<T>>,
    /// Whether to add zero key/value for incremental decoding
    pub add_zero_attn: bool,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> MultiHeadAttention<T> {
    /// Create a new MultiHeadAttention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of parallel attention heads
    /// * `dropout` - Dropout probability (0.0 = no dropout)
    /// * `bias` - Whether to add bias to the projections
    /// * `add_bias_kv` - Whether to add bias to key/value projections
    /// * `add_zero_attn` - Whether to add a zero attention for incremental decoding
    /// * `kdim` - Override for key embedding dimension (for cross-attention)
    /// * `vdim` - Override for value embedding dimension (for cross-attention)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: f64,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
    ) -> Self {
        assert_eq!(
            embed_dim % num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );

        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);

        let q_proj = crate::Linear::new(embed_dim, embed_dim);
        let k_proj = crate::Linear::new(kdim, embed_dim);
        let v_proj = crate::Linear::new(vdim, embed_dim);
        let out_proj = crate::Linear::new(embed_dim, embed_dim);

        let add_bias_kv_tensor = if add_bias_kv {
            Some(Tensor::zeros(vec![1, 1, embed_dim]))
        } else {
            None
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            embed_dim,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv: add_bias_kv_tensor,
            add_zero_attn,
        }
    }

    /// Forward pass for multi-head attention
    ///
    /// # Arguments
    /// * `query` - Query tensor (batch_size, tgt_len, embed_dim)
    /// * `key` - Key tensor (batch_size, src_len, embed_dim)
    /// * `value` - Value tensor (batch_size, src_len, embed_dim)
    /// * `key_padding_mask` - Optional mask for padded positions in key
    /// * `need_weights` - Whether to return attention weights
    /// * `attn_mask` - Optional attention mask
    /// * `average_attn_weights` - Whether to average attention weights across heads
    ///
    /// # Returns
    /// Tuple of (output, attention_weights) where attention_weights is optional
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mha(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        key_padding_mask: Option<&Tensor<T>>,
        need_weights: bool,
        attn_mask: Option<&Tensor<T>>,
        average_attn_weights: bool,
    ) -> crate::Result<(Tensor<T>, Option<Tensor<T>>)> {
        let tgt_len = query.shape()[1];
        let src_len = key.shape()[1];
        let batch_size = query.shape()[0];

        // Project query, key, value
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape for multi-head attention
        let head_dim = self.embed_dim / self.num_heads;
        let q = self._reshape_for_attention(&q, batch_size, tgt_len, self.num_heads, head_dim)?;
        let k = self._reshape_for_attention(&k, batch_size, src_len, self.num_heads, head_dim)?;
        let v = self._reshape_for_attention(&v, batch_size, src_len, self.num_heads, head_dim)?;

        // Scaled dot-product attention
        let (attn_output, attn_weights) = self._scaled_dot_product_attention(
            &q,
            &k,
            &v,
            attn_mask,
            key_padding_mask,
            need_weights,
        )?;

        // Reshape back and apply output projection
        let attn_output =
            self._reshape_from_attention(&attn_output, batch_size, tgt_len, self.embed_dim)?;
        let output = self.out_proj.forward(&attn_output)?;

        // Handle attention weights
        let attn_weights = if need_weights {
            if let Some(weights) = attn_weights {
                if average_attn_weights {
                    // Average across heads: (batch, num_heads, tgt_len, src_len) -> (batch, tgt_len, src_len)
                    Some(weights.mean_dim(Some(1), true)?)
                } else {
                    Some(weights)
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok((output, attn_weights))
    }

    /// Reshape tensor for multi-head attention computation
    fn _reshape_for_attention(
        &self,
        tensor: &Tensor<T>,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<Tensor<T>> {
        // Input: (batch_size, seq_len, embed_dim)
        // Output: (batch_size, num_heads, seq_len, head_dim)

        let mut reshaped_data = Vec::with_capacity(tensor.data().len());

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let src_idx =
                            b * seq_len * self.embed_dim + s * self.embed_dim + h * head_dim + d;
                        reshaped_data.push(tensor.data()[src_idx]);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            reshaped_data,
            vec![batch_size, num_heads, seq_len, head_dim],
        ))
    }

    /// Reshape tensor back from multi-head attention computation
    fn _reshape_from_attention(
        &self,
        tensor: &Tensor<T>,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
    ) -> crate::Result<Tensor<T>> {
        // Input: (batch_size, num_heads, seq_len, head_dim)
        // Output: (batch_size, seq_len, embed_dim)

        let mut reshaped_data = Vec::with_capacity(tensor.data().len());

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..(embed_dim / self.num_heads) {
                        let src_idx = b * self.num_heads * seq_len * (embed_dim / self.num_heads)
                            + h * seq_len * (embed_dim / self.num_heads)
                            + s * (embed_dim / self.num_heads)
                            + d;
                        reshaped_data.push(tensor.data()[src_idx]);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            reshaped_data,
            vec![batch_size, seq_len, embed_dim],
        ))
    }

    /// Scaled dot-product attention implementation
    fn _scaled_dot_product_attention(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        attn_mask: Option<&Tensor<T>>,
        key_padding_mask: Option<&Tensor<T>>,
        need_weights: bool,
    ) -> crate::Result<(Tensor<T>, Option<Tensor<T>>)> {
        let batch_size = query.shape()[0];
        let num_heads = query.shape()[1];
        let tgt_len = query.shape()[2];
        let src_len = key.shape()[2];
        let head_dim = query.shape()[3];

        // Compute attention scores: (batch, num_heads, tgt_len, src_len)
        let mut attn_scores = Vec::with_capacity(batch_size * num_heads * tgt_len * src_len);

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..tgt_len {
                    for j in 0..src_len {
                        let mut score = T::zero();

                        // Dot product of query and key
                        for d in 0..head_dim {
                            let q_idx = ((b * num_heads + h) * tgt_len + i) * head_dim + d;
                            let k_idx = ((b * num_heads + h) * src_len + j) * head_dim + d;
                            score = score + query.data()[q_idx] * key.data()[k_idx];
                        }

                        // Scale by sqrt(head_dim)
                        score = score / T::from((head_dim as f64).sqrt()).unwrap();
                        attn_scores.push(score);
                    }
                }
            }
        }

        // Apply attention mask if provided
        if let Some(mask) = attn_mask {
            for (i, score) in attn_scores.iter_mut().enumerate() {
                let mask_val = mask.data()[i % mask.data().len()];
                if mask_val == T::zero() {
                    // In PyTorch, mask values of 0 indicate positions to mask
                    *score = T::neg_infinity();
                }
            }
        }

        // Apply key padding mask if provided
        if let Some(padding_mask) = key_padding_mask {
            for b in 0..batch_size {
                for h in 0..num_heads {
                    for i in 0..tgt_len {
                        for j in 0..src_len {
                            let mask_idx = b * src_len + j;
                            let mask_val = padding_mask.data()[mask_idx];

                            if mask_val != T::zero() {
                                let score_idx = ((b * num_heads + h) * tgt_len + i) * src_len + j;
                                attn_scores[score_idx] = T::neg_infinity();
                            }
                        }
                    }
                }
            }
        }

        // Compute softmax over attention scores
        let mut softmax_scores = Vec::with_capacity(attn_scores.len());
        let scores_per_head = tgt_len * src_len;

        for head_start in (0..attn_scores.len()).step_by(scores_per_head) {
            let head_end = (head_start + scores_per_head).min(attn_scores.len());
            let head_scores = &attn_scores[head_start..head_end];

            // Find max for numerical stability
            let mut max_score = T::neg_infinity();
            for &score in head_scores {
                if score > max_score && score != T::neg_infinity() {
                    max_score = score;
                }
            }

            // Compute softmax
            let mut exp_sum = T::zero();
            for &score in head_scores {
                let exp_score = if score == T::neg_infinity() {
                    T::zero()
                } else {
                    (score - max_score).exp()
                };
                softmax_scores.push(exp_score);
                exp_sum = exp_sum + exp_score;
            }

            // Normalize
            for i in 0..head_scores.len() {
                softmax_scores[head_start + i] = softmax_scores[head_start + i] / exp_sum;
            }
        }

        // Compute attention output: softmax_scores @ value
        let mut output_data = Vec::with_capacity(batch_size * num_heads * tgt_len * head_dim);

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..tgt_len {
                    for d in 0..head_dim {
                        let mut sum = T::zero();

                        for j in 0..src_len {
                            let attn_idx = ((b * num_heads + h) * tgt_len + i) * src_len + j;
                            let v_idx = ((b * num_heads + h) * src_len + j) * head_dim + d;
                            sum = sum + softmax_scores[attn_idx] * value.data()[v_idx];
                        }

                        output_data.push(sum);
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, vec![batch_size, num_heads, tgt_len, head_dim]);

        let attn_weights = if need_weights {
            Some(Tensor::from_vec(
                softmax_scores,
                vec![batch_size, num_heads, tgt_len, src_len],
            ))
        } else {
            None
        };

        Ok((output, attn_weights))
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T>
    for MultiHeadAttention<T>
{
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For self-attention, query = key = value = input
        let (output, _) = self
            .forward_mha(input, input, input, None, false, None, true)
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("MultiHeadAttention forward failed: {:?}", e),
            })?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
    }
}

/// Causal Self-Attention layer (GPT-2 style)
#[derive(Debug, Clone)]
pub struct CausalSelfAttention<T: FloatDtype> {
    /// Query, Key, Value projection matrix (combined)
    pub c_attn: crate::Linear<T>,
    /// Output projection matrix
    pub c_proj: crate::Linear<T>,
    /// Attention configuration
    pub config: AttentionConfig,
    /// Causal mask for autoregressive attention
    pub bias: Option<Tensor<T>>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> CausalSelfAttention<T> {
    /// Create a new causal self-attention layer
    ///
    /// # Arguments
    /// * `config` - Attention configuration
    pub fn new(config: AttentionConfig) -> Self {
        // Validate configuration
        assert_eq!(
            config.n_embd % config.n_head,
            0,
            "Embedding dimension must be divisible by number of heads"
        );

        // Combined QKV projection: 3 * n_embd for Q, K, V
        let c_attn = crate::Linear::new(config.n_embd, 3 * config.n_embd);
        let c_proj = crate::Linear::new(config.n_embd, config.n_embd);

        // Create causal mask if needed
        let bias = if config.causal {
            // Lower triangular mask of shape (1, 1, block_size, block_size)
            let mut bias_data = vec![T::zero(); config.block_size * config.block_size];

            for i in 0..config.block_size {
                for j in 0..config.block_size {
                    if j > i {
                        // Set to -inf for positions that should be masked
                        bias_data[i * config.block_size + j] = T::neg_infinity();
                    }
                }
            }

            Some(Tensor::from_vec(
                bias_data,
                vec![1, 1, config.block_size, config.block_size],
            ))
        } else {
            None
        };

        Self {
            c_attn,
            c_proj,
            config,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for CausalSelfAttention<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // input shape: (batch_size, seq_len, n_embd)
        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![0, 0, self.config.n_embd],
                actual: input_shape.to_vec(),
            });
        }

        let (batch_size, seq_len, _n_embd) = (input_shape[0], input_shape[1], input_shape[2]);

        if seq_len > self.config.block_size {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Sequence length {} exceeds block size {}",
                    seq_len, self.config.block_size
                ),
            });
        }

        // Compute Q, K, V projections
        let qkv = self.c_attn.forward(input)?;
        // qkv shape: (batch_size, seq_len, 3 * n_embd)

        // Split into Q, K, V
        let total_features = 3 * self.config.n_embd;
        let head_size = self.config.n_embd / self.config.n_head;

        // Reshape for multi-head attention
        // From (batch_size, seq_len, 3 * n_embd) to (batch_size, seq_len, n_head, 3 * head_size)
        let mut qkv_reshaped = Vec::with_capacity(qkv.data().len());

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.config.n_head {
                    for f in 0..(3 * head_size) {
                        let src_idx = b * seq_len * total_features
                            + s * total_features
                            + h * head_size * 3
                            + f;
                        qkv_reshaped.push(qkv.data()[src_idx]);
                    }
                }
            }
        }

        // Split into Q, K, V
        let mut q_data = Vec::with_capacity(batch_size * seq_len * self.config.n_head * head_size);
        let mut k_data = Vec::with_capacity(batch_size * seq_len * self.config.n_head * head_size);
        let mut v_data = Vec::with_capacity(batch_size * seq_len * self.config.n_head * head_size);

        for i in 0..qkv_reshaped.len() / 3 {
            q_data.push(qkv_reshaped[i * 3]);
            k_data.push(qkv_reshaped[i * 3 + 1]);
            v_data.push(qkv_reshaped[i * 3 + 2]);
        }

        // Create Q, K, V tensors with shape (batch_size, n_head, seq_len, head_size)
        let q_shape = vec![batch_size, self.config.n_head, seq_len, head_size];
        let k_shape = vec![batch_size, self.config.n_head, seq_len, head_size];
        let v_shape = vec![batch_size, self.config.n_head, seq_len, head_size];

        let q = Tensor::from_vec(q_data, q_shape);
        let k = Tensor::from_vec(k_data, k_shape);
        let v = Tensor::from_vec(v_data, v_shape);

        // Attention computation
        // att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(head_size))

        // For simplicity, implement attention for a single head and batch first
        // In a full implementation, this would handle batched multi-head attention

        // Compute attention scores
        let mut att_scores = Vec::new();
        for b in 0..batch_size {
            for h in 0..self.config.n_head {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score = T::zero();
                        for d in 0..head_size {
                            let q_idx =
                                ((b * self.config.n_head + h) * seq_len + i) * head_size + d;
                            let k_idx =
                                ((b * self.config.n_head + h) * seq_len + j) * head_size + d;
                            score = score + q.data()[q_idx] * k.data()[k_idx];
                        }
                        score = score / T::from((head_size as f64).sqrt()).unwrap();
                        att_scores.push(score);
                    }
                }
            }
        }

        // Apply causal mask if needed
        if let Some(ref mask) = self.bias {
            #[allow(clippy::needless_range_loop)]
            for i in 0..att_scores.len() {
                let mask_val = mask.data()[i % mask.data().len()];
                if mask_val == T::neg_infinity() {
                    att_scores[i] = T::neg_infinity();
                }
            }
        }

        // Apply softmax to attention scores
        let mut softmax_scores = Vec::new();
        let scores_per_head = seq_len * seq_len;

        for head_start in (0..att_scores.len()).step_by(scores_per_head) {
            let head_end = (head_start + scores_per_head).min(att_scores.len());
            let head_scores = &att_scores[head_start..head_end];

            // Find max for numerical stability
            let mut max_score = T::neg_infinity();
            for &score in head_scores {
                if score > max_score {
                    max_score = score;
                }
            }

            // Compute softmax
            let mut exp_sum = T::zero();
            for &score in head_scores {
                let exp_score = (score - max_score).exp();
                softmax_scores.push(exp_score);
                exp_sum = exp_sum + exp_score;
            }

            // Normalize
            for i in 0..head_scores.len() {
                softmax_scores[head_start + i] = softmax_scores[head_start + i] / exp_sum;
            }
        }

        // Compute attention output: softmax_scores @ v
        let mut output_data = Vec::with_capacity(batch_size * seq_len * self.config.n_embd);

        for b in 0..batch_size {
            for h in 0..self.config.n_head {
                for i in 0..seq_len {
                    for d in 0..head_size {
                        let mut sum = T::zero();
                        for j in 0..seq_len {
                            let att_idx =
                                ((b * self.config.n_head + h) * seq_len + i) * seq_len + j;
                            let v_idx =
                                ((b * self.config.n_head + h) * seq_len + j) * head_size + d;
                            sum = sum + softmax_scores[att_idx] * v.data()[v_idx];
                        }
                        output_data.push(sum);
                    }
                }
            }
        }

        // Reshape back to (batch_size, seq_len, n_embd)
        let output_shape = vec![batch_size, seq_len, self.config.n_embd];
        let output = Tensor::from_vec(output_data, output_shape);

        // Apply output projection
        let output = self.c_proj.forward(&output)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.c_attn.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.c_attn.parameters_mut());
        params.extend(self.c_proj.parameters_mut());
        params
    }
}

impl<T: FloatDtype> fmt::Display for CausalSelfAttention<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CausalSelfAttention(n_head={}, n_embd={}, block_size={})",
            self.config.n_head, self.config.n_embd, self.config.block_size
        )
    }
}

/// MLP block for transformer layers
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // MLP: input -> Linear -> GELU -> Linear -> output
        let hidden = self.c_fc.forward(input)?;
        let activated = self.gelu.forward(&hidden)?;
        let output = self.c_proj.forward(&activated)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
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

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
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

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
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

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
        tgt: &Tensor<T>,
        memory: Option<&Tensor<T>>,
        tgt_mask: Option<&Tensor<T>>,
        memory_mask: Option<&Tensor<T>>,
    ) -> crate::Result<Tensor<T>> {
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
            let output = (&residual2 + &ff_output)?;

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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, assume self-attention only (no cross-attention)
        self.forward(input, None, None, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
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

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
        tgt: &Tensor<T>,
        memory: Option<&Tensor<T>>,
        tgt_mask: Option<&Tensor<T>>,
        memory_mask: Option<&Tensor<T>>,
    ) -> crate::Result<Tensor<T>> {
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, assume self-attention only
        self.forward(input, None, None, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
        src: &Tensor<T>,
        tgt: &Tensor<T>,
        _src_mask: Option<&Tensor<T>>,
        tgt_mask: Option<&Tensor<T>>,
        memory_mask: Option<&Tensor<T>>,
    ) -> crate::Result<Tensor<T>> {
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, assume encoder-only operation (no decoder)
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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

    #[test]
    fn test_causal_attention_creation() {
        let config = AttentionConfig::default();
        let attention: CausalSelfAttention<f32> = CausalSelfAttention::new(config.clone());

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
        let attention: CausalSelfAttention<f32> = CausalSelfAttention::new(config);

        // Input shape: (batch_size=1, seq_len=3, n_embd=8)
        let input = Tensor::from_vec(vec![1.0; 24], vec![1, 3, 8]);
        let output = attention.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_mlp_creation() {
        let mlp: MLP<f32> = MLP::new(768);

        assert_eq!(mlp.c_fc.in_features, 768);
        assert_eq!(mlp.c_fc.out_features, 3072); // 4x expansion
        assert_eq!(mlp.c_proj.in_features, 3072);
        assert_eq!(mlp.c_proj.out_features, 768);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp: MLP<f32> = MLP::new(8);

        // Input shape: (batch_size=1, seq_len=3, n_embd=8)
        let input = Tensor::from_vec(vec![1.0; 24], vec![1, 3, 8]);
        let output = mlp.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_multihead_attention_creation() {
        let mha: MultiHeadAttention<f32> = MultiHeadAttention::new(
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
        let mha: MultiHeadAttention<f32> = MultiHeadAttention::new(
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
        let input = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);
        let (output, weights) = mha
            .forward_mha(&input, &input, &input, None, true, None, false)
            .unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
        // Attention weights should be (batch_size, num_heads, seq_len, seq_len)
        assert_eq!(weights.as_ref().unwrap().shape(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_multihead_attention_cross_attention() {
        let mha: MultiHeadAttention<f32> = MultiHeadAttention::new(
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
        let query = Tensor::from_vec(vec![1.0; 384], vec![2, 3, 64]);
        // Key/Value: (batch_size=2, src_len=5, kdim/vdim=32)
        let key = Tensor::from_vec(vec![1.0; 320], vec![2, 5, 32]);
        let value = Tensor::from_vec(vec![1.0; 320], vec![2, 5, 32]);

        let (output, weights) = mha
            .forward_mha(&query, &key, &value, None, true, None, false)
            .unwrap();

        // Output should match query dimensions
        assert_eq!(output.shape(), &[2, 3, 64]);
        // Attention weights should be (batch_size, num_heads, tgt_len, src_len)
        assert_eq!(weights.as_ref().unwrap().shape(), &[2, 8, 3, 5]);
    }

    #[test]
    fn test_multihead_attention_module_trait() {
        let mha: MultiHeadAttention<f32> = MultiHeadAttention::new(
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
        let input = Tensor::from_vec(vec![1.0; 96], vec![1, 3, 32]);
        let output = mha.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 3, 32]);

        // Check parameters: each Linear layer has weight and bias (when bias=true)
        let params = mha.parameters();
        assert_eq!(params.len(), 8); // q_proj, k_proj, v_proj, out_proj (weight + bias each)
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let encoder_layer: TransformerEncoderLayer<f32> = TransformerEncoderLayer::new(
            64,  // d_model
            8,   // nhead
            128, // dim_feedforward
            0.1, // dropout
        );

        // Input: (batch_size=2, seq_len=4, d_model=64)
        let input = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);
        let output = encoder_layer.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let encoder_layer: TransformerEncoderLayer<f32> =
            TransformerEncoderLayer::new(64, 8, 128, 0.1);
        let encoder: TransformerEncoder<f32> = TransformerEncoder::new(encoder_layer, 2, None);

        // Input: (batch_size=2, seq_len=4, d_model=64)
        let input = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);
        let output = encoder.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4, 64]);
        assert_eq!(encoder.layers.len(), 2);
    }

    #[test]
    fn test_transformer_decoder_layer() {
        let decoder_layer: TransformerDecoderLayer<f32> = TransformerDecoderLayer::new(
            64,  // d_model
            8,   // nhead
            128, // dim_feedforward
            0.1, // dropout
        );

        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = Tensor::from_vec(vec![1.0; 384], vec![2, 3, 64]);
        // Memory (encoder output): (batch_size=2, src_len=4, d_model=64)
        let memory = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);

        let output = decoder_layer
            .forward(&tgt, Some(&memory), None, None)
            .unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
    }

    #[test]
    fn test_transformer_decoder() {
        let decoder_layer: TransformerDecoderLayer<f32> =
            TransformerDecoderLayer::new(64, 8, 128, 0.1);
        let decoder: TransformerDecoder<f32> = TransformerDecoder::new(decoder_layer, 2, None);

        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = Tensor::from_vec(vec![1.0; 384], vec![2, 3, 64]);
        // Memory (encoder output): (batch_size=2, src_len=4, d_model=64)
        let memory = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);

        let output = decoder.forward(&tgt, Some(&memory), None, None).unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
        assert_eq!(decoder.layers.len(), 2);
    }

    #[test]
    fn test_transformer() {
        let transformer: Transformer<f32> = Transformer::new(
            64,  // d_model
            8,   // nhead
            2,   // num_encoder_layers
            2,   // num_decoder_layers
            128, // dim_feedforward
            0.1, // dropout
        );

        // Source input: (batch_size=2, src_len=4, d_model=64)
        let src = Tensor::from_vec(vec![1.0; 512], vec![2, 4, 64]);
        // Target input: (batch_size=2, tgt_len=3, d_model=64)
        let tgt = Tensor::from_vec(vec![1.0; 384], vec![2, 3, 64]);

        let output = transformer.forward(&src, &tgt, None, None, None).unwrap();

        // Output should have same shape as target
        assert_eq!(output.shape(), &[2, 3, 64]);
        assert_eq!(transformer.encoder.layers.len(), 2);
        assert_eq!(transformer.decoder.layers.len(), 2);
    }
}
