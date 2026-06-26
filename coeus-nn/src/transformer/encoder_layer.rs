// ── Transformer Encoder Layer (Pre-LayerNorm) ──

use super::ffn::FeedForward;
use crate::attention::{MultiHeadAttention, NullMask};
use crate::dropout::Dropout;
use crate::module::Module;
use crate::normalization::LayerNorm;
use coeus_autograd::{AttentionMask, Var};
use coeus_core::{Float, MoiraiBackend};

/// Single Transformer encoder layer.
///
/// Pre-LayerNorm sub-layer composition:
/// ```text
///   x₁ = x + Dropout(SelfAttn(LayerNorm(x)))
///   x₂ = x₁ + Dropout(FFN(LayerNorm(x₁)))
/// ```
pub struct TransformerEncoderLayer<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    M: AttentionMask = NullMask,
> {
    pub norm1: LayerNorm<T, B>,
    pub self_attn: MultiHeadAttention<T, B, H, M>,
    pub dropout1: Dropout,
    pub norm2: LayerNorm<T, B>,
    pub ffn: FeedForward<T, B>,
    pub dropout2: Dropout,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const H: usize, M: AttentionMask>
    TransformerEncoderLayer<T, B, H, M>
{
    /// Construct an encoder layer.
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        Self {
            norm1: LayerNorm::new(d_model, 1e-5),
            self_attn: MultiHeadAttention::new(d_model, true),
            dropout1: Dropout::new(dropout_p),
            norm2: LayerNorm::new(d_model, 1e-5),
            ffn: FeedForward::new(d_model, d_ff, dropout_p),
            dropout2: Dropout::new(dropout_p),
        }
    }

    /// Apply LayerNorm to a 3D `[batch, seq, d_model]` var.
    ///
    /// Uses tracked reshape (`coeus_autograd::reshape`) to flatten to `[batch*seq, d_model]`,
    /// apply LayerNorm, and reshape back. Gradients flow through all three ops.
    fn layernorm_3d(&self, norm: &LayerNorm<T, B>, x: &Var<T, B>) -> Var<T, B> {
        let shape = x.tensor.shape_cloned();
        let batch = shape[0];
        let seq = shape[1];
        let d = shape[2];
        // Tracked flatten [batch, seq, d] → [batch*seq, d]
        let flat = coeus_autograd::reshape(x, [batch * seq, d]);
        // Apply LayerNorm ([batch*seq, d] → [batch*seq, d])
        let normed = norm.forward(&flat);
        // Tracked unflatten [batch*seq, d] → [batch, seq, d]
        coeus_autograd::reshape(&normed, [batch, seq, d])
    }

    /// Pre-LayerNorm forward with optional key padding mask.
    ///
    /// Input/output shape: `[batch, seq, d_model]`.
    /// `key_padding_mask` shape: `[batch, seq]` (or backend-supported broadcast form).
    pub fn forward_with_mask(
        &self,
        input: &Var<T, B>,
        key_padding_mask: Option<&Var<T, B>>,
    ) -> Var<T, B> {
        // Sub-layer 1: Self-Attention with residual
        let normed1 = self.layernorm_3d(&self.norm1, input);
        let attn_out = self
            .self_attn
            .forward_cross(&normed1, &normed1, &normed1, key_padding_mask);
        let dropped1 = self.dropout1.forward(&attn_out);
        let x = coeus_autograd::add(input, &dropped1);

        // Sub-layer 2: FFN with residual
        let normed2 = self.layernorm_3d(&self.norm2, &x);
        let ffn_out = self.ffn.forward(&normed2);
        let dropped2 = self.dropout2.forward(&ffn_out);
        coeus_autograd::add(&x, &dropped2)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const H: usize, M: AttentionMask> Module<T, B>
    for TransformerEncoderLayer<T, B, H, M>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.norm1.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.ffn.parameters());
        p
    }

    /// Pre-LayerNorm forward. Input/output: `[batch, seq, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward_with_mask(input, None)
    }
}

/// Manual Clone impl.
impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const H: usize, M: AttentionMask> Clone
    for TransformerEncoderLayer<T, B, H, M>
where
    LayerNorm<T, B>: Clone,
    MultiHeadAttention<T, B, H, M>: Clone,
    FeedForward<T, B>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            norm1: self.norm1.clone(),
            self_attn: self.self_attn.clone(),
            dropout1: self.dropout1.clone(),
            norm2: self.norm2.clone(),
            ffn: self.ffn.clone(),
            dropout2: self.dropout2.clone(),
        }
    }
}
