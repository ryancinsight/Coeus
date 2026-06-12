// ── Transformer Decoder Layer (Pre-LayerNorm) ──

use super::ffn::FeedForward;
use crate::attention::MultiHeadAttention;
use crate::dropout::Dropout;
use crate::module::Module;
use crate::normalization::LayerNorm;
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::{Float, MoiraiBackend};

/// Single Transformer decoder layer.
///
/// Pre-LayerNorm sub-layer composition:
/// ```text
///   x₁ = x + Dropout(SelfAttn(LayerNorm(x)))
///   x₂ = x₁ + Dropout(CrossAttn(LayerNorm(x₁), memory, memory))
///   x₃ = x₂ + Dropout(FFN(LayerNorm(x₂)))
/// ```
pub struct TransformerDecoderLayer<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    SelfM: AttentionMask = CausalMask,
    CrossM: AttentionMask = NullMask,
> {
    pub norm1: LayerNorm<T, B>,
    pub self_attn: MultiHeadAttention<T, B, H, SelfM>,
    pub dropout1: Dropout,
    pub norm2: LayerNorm<T, B>,
    pub cross_attn: MultiHeadAttention<T, B, H, CrossM>,
    pub dropout2: Dropout,
    pub norm3: LayerNorm<T, B>,
    pub ffn: FeedForward<T, B>,
    pub dropout3: Dropout,
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        SelfM: AttentionMask,
        CrossM: AttentionMask,
    > TransformerDecoderLayer<T, B, H, SelfM, CrossM>
{
    /// Construct a decoder layer.
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        Self {
            norm1: LayerNorm::new(d_model, 1e-5),
            self_attn: MultiHeadAttention::new(d_model, true),
            dropout1: Dropout::new(dropout_p),
            norm2: LayerNorm::new(d_model, 1e-5),
            cross_attn: MultiHeadAttention::new(d_model, true),
            dropout2: Dropout::new(dropout_p),
            norm3: LayerNorm::new(d_model, 1e-5),
            ffn: FeedForward::new(d_model, d_ff, dropout_p),
            dropout3: Dropout::new(dropout_p),
        }
    }

    /// Apply LayerNorm to a 3D `[batch, seq, d_model]` var.
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

    /// Decode forward pass with cross-attention.
    ///
    /// - `tgt`:    target inputs `[batch, seq_tgt, d_model]`
    /// - `memory`: encoder outputs `[batch, seq_src, d_model]`
    pub fn forward_decoder(&self, tgt: &Var<T, B>, memory: &Var<T, B>) -> Var<T, B> {
        // Sub-layer 1: Masked Self-Attention with residual
        let normed1 = self.layernorm_3d(&self.norm1, tgt);
        let self_attn_out = self.self_attn.forward(&normed1);
        let dropped1 = self.dropout1.forward(&self_attn_out);
        let x = coeus_autograd::add(tgt, &dropped1);

        // Sub-layer 2: Cross-Attention with residual
        let normed2 = self.layernorm_3d(&self.norm2, &x);
        let cross_attn_out = self
            .cross_attn
            .forward_cross(&normed2, memory, memory, None);
        let dropped2 = self.dropout2.forward(&cross_attn_out);
        let x = coeus_autograd::add(&x, &dropped2);

        // Sub-layer 3: FFN with residual
        let normed3 = self.layernorm_3d(&self.norm3, &x);
        let ffn_out = self.ffn.forward(&normed3);
        let dropped3 = self.dropout3.forward(&ffn_out);
        coeus_autograd::add(&x, &dropped3)
    }
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        SelfM: AttentionMask,
        CrossM: AttentionMask,
    > Module<T, B> for TransformerDecoderLayer<T, B, H, SelfM, CrossM>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.norm1.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.cross_attn.parameters());
        p.extend(self.norm3.parameters());
        p.extend(self.ffn.parameters());
        p
    }

    /// Fallback forward without cross-attention: `memory = tgt`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward_decoder(input, input)
    }
}

/// Manual Clone impl.
impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        SelfM: AttentionMask,
        CrossM: AttentionMask,
    > Clone for TransformerDecoderLayer<T, B, H, SelfM, CrossM>
where
    LayerNorm<T, B>: Clone,
    MultiHeadAttention<T, B, H, SelfM>: Clone,
    MultiHeadAttention<T, B, H, CrossM>: Clone,
    FeedForward<T, B>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            norm1: self.norm1.clone(),
            self_attn: self.self_attn.clone(),
            dropout1: self.dropout1.clone(),
            norm2: self.norm2.clone(),
            cross_attn: self.cross_attn.clone(),
            dropout2: self.dropout2.clone(),
            norm3: self.norm3.clone(),
            ffn: self.ffn.clone(),
            dropout3: self.dropout3.clone(),
        }
    }
}
