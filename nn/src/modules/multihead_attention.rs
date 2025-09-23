//! Multi-Head Attention layer
//!
//! Implements the standard multi-head attention mechanism used in transformers.
//! Supports both self-attention and cross-attention modes.

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};

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
    ) -> Result<(Tensor<T>, Option<Tensor<T>>)> {
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
    ) -> Result<Tensor<T>> {
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
    ) -> Result<Tensor<T>> {
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
    ) -> Result<(Tensor<T>, Option<Tensor<T>>)> {
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
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // For self-attention, query = key = value = input
        let (output, _) = self
            .forward_mha(input, input, input, None, false, None, true)
            .map_err(|e| NNError::InvalidInput {
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
