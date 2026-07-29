// ── Transformer Decoder Layer (Pre-LayerNorm) ──

use super::ffn::{feed_forward_with_training, FeedForward};
use crate::attention::{multi_head_attention_cross, MhaProjectionParams, MultiHeadAttention};
use crate::dropout::Dropout;
use crate::module::{prefixed_parameters, Module, ModuleError};
use crate::normalization::LayerNorm;
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::{Float, MoiraiBackend};

/// Borrowed parameters for functional transformer decoder-layer execution.
pub struct TransformerDecoderLayerParams<'a, T: Float, B: coeus_ops::BackendOps<T> + Default> {
    /// LayerNorm1 gamma `[d_model]`.
    pub norm1_weight: &'a Var<T, B>,
    /// LayerNorm1 beta `[d_model]`.
    pub norm1_bias: &'a Var<T, B>,
    /// Masked self-attention projections.
    pub self_attn: MhaProjectionParams<'a, T, B>,
    /// LayerNorm2 gamma `[d_model]`.
    pub norm2_weight: &'a Var<T, B>,
    /// LayerNorm2 beta `[d_model]`.
    pub norm2_bias: &'a Var<T, B>,
    /// Cross-attention projections.
    pub cross_attn: MhaProjectionParams<'a, T, B>,
    /// LayerNorm3 gamma `[d_model]`.
    pub norm3_weight: &'a Var<T, B>,
    /// LayerNorm3 beta `[d_model]`.
    pub norm3_bias: &'a Var<T, B>,
    /// FFN linear1 weight `[d_ff, d_model]`.
    pub ffn_w1: &'a Var<T, B>,
    /// FFN linear1 bias `[d_ff]`.
    pub ffn_b1: Option<&'a Var<T, B>>,
    /// FFN linear2 weight `[d_model, d_ff]`.
    pub ffn_w2: &'a Var<T, B>,
    /// FFN linear2 bias `[d_model]`.
    pub ffn_b2: Option<&'a Var<T, B>>,
    /// Dropout probability after masked self-attention.
    pub self_attn_residual_dropout_p: f64,
    /// Training mode for self-attention residual dropout.
    pub self_attn_residual_training: bool,
    /// Dropout probability after cross-attention.
    pub cross_attn_residual_dropout_p: f64,
    /// Training mode for cross-attention residual dropout.
    pub cross_attn_residual_training: bool,
    /// Dropout probability inside FFN hidden projection.
    pub ffn_hidden_dropout_p: f64,
    /// Training mode for FFN hidden dropout.
    pub ffn_hidden_training: bool,
    /// Dropout probability after FFN residual branch.
    pub ffn_residual_dropout_p: f64,
    /// Training mode for FFN residual dropout.
    pub ffn_residual_training: bool,
}

fn apply_dropout<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    p: f64,
    is_training: bool,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    let mut dropout = Dropout::new(p);
    dropout.set_training(is_training);
    dropout.forward(x)
}

/// Functional (stateless) TransformerDecoderLayer forward.
///
/// Computes pre-LN decoder sublayers:
/// `x1 = tgt + Dropout(SelfAttn(LN1(tgt)))`
/// `x2 = x1 + Dropout(CrossAttn(LN2(x1), memory, memory))`
/// `x3 = x2 + Dropout(Linear2(Dropout(GELU(Linear1(LN3(x2))))))`.
pub fn transformer_decoder_layer<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const H: usize,
    SelfM: AttentionMask,
    CrossM: AttentionMask,
>(
    tgt: &Var<T, B>,
    memory: &Var<T, B>,
    params: TransformerDecoderLayerParams<'_, T, B>,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    const MODULE: &str = "TransformerDecoderLayer";
    let ([_, _, d_model], _) = super::validation::decoder_inputs(MODULE, tgt, memory)?;
    for (parameter, value) in [
        ("norm1 weight", params.norm1_weight),
        ("norm1 bias", params.norm1_bias),
        ("norm2 weight", params.norm2_weight),
        ("norm2 bias", params.norm2_bias),
        ("norm3 weight", params.norm3_weight),
        ("norm3 bias", params.norm3_bias),
    ] {
        super::validation::affine_vector(MODULE, parameter, value, d_model)?;
    }
    super::validation::feed_forward(
        MODULE,
        d_model,
        params.ffn_w1,
        params.ffn_b1,
        params.ffn_w2,
        params.ffn_b2,
    )?;
    let normed1 = super::normalization::layer_norm_three_dimensional(
        MODULE,
        tgt,
        params.norm1_weight,
        params.norm1_bias,
        1e-5,
    )?;
    let self_attn_out = multi_head_attention_cross::<T, B, H, SelfM>(
        &normed1,
        &normed1,
        &normed1,
        params.self_attn,
        None,
    )?;
    let dropped1 = apply_dropout(
        &self_attn_out,
        params.self_attn_residual_dropout_p,
        params.self_attn_residual_training,
    )?;
    let x = coeus_autograd::add(tgt, &dropped1);

    let normed2 = super::normalization::layer_norm_three_dimensional(
        MODULE,
        &x,
        params.norm2_weight,
        params.norm2_bias,
        1e-5,
    )?;
    let cross_attn_out = multi_head_attention_cross::<T, B, H, CrossM>(
        &normed2,
        memory,
        memory,
        params.cross_attn,
        None,
    )?;
    let dropped2 = apply_dropout(
        &cross_attn_out,
        params.cross_attn_residual_dropout_p,
        params.cross_attn_residual_training,
    )?;
    let x = coeus_autograd::add(&x, &dropped2);

    let normed3 = super::normalization::layer_norm_three_dimensional(
        MODULE,
        &x,
        params.norm3_weight,
        params.norm3_bias,
        1e-5,
    )?;
    let ff_out = feed_forward_with_training(
        &normed3,
        params.ffn_w1,
        params.ffn_b1,
        params.ffn_w2,
        params.ffn_b2,
        params.ffn_hidden_dropout_p,
        params.ffn_hidden_training,
    )?;
    let dropped3 = apply_dropout(
        &ff_out,
        params.ffn_residual_dropout_p,
        params.ffn_residual_training,
    )?;
    Ok(coeus_autograd::add(&x, &dropped3))
}

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
    /// Pre-LayerNorm before self-attention.
    pub norm1: LayerNorm<T, B>,
    /// Masked self-attention sub-layer.
    pub self_attn: MultiHeadAttention<T, B, H, SelfM>,
    /// Dropout applied after self-attention.
    pub dropout1: Dropout,
    /// Pre-LayerNorm before cross-attention.
    pub norm2: LayerNorm<T, B>,
    /// Cross-attention sub-layer attending to encoder memory.
    pub cross_attn: MultiHeadAttention<T, B, H, CrossM>,
    /// Dropout applied after cross-attention.
    pub dropout2: Dropout,
    /// Pre-LayerNorm before FFN.
    pub norm3: LayerNorm<T, B>,
    /// Feed-forward sub-layer.
    pub ffn: FeedForward<T, B>,
    /// Dropout applied after FFN.
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

    /// Decode forward pass with cross-attention.
    ///
    /// - `tgt`:    target inputs `[batch, seq_tgt, d_model]`
    /// - `memory`: encoder outputs `[batch, seq_src, d_model]`
    pub fn forward_decoder(
        &self,
        tgt: &Var<T, B>,
        memory: &Var<T, B>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        transformer_decoder_layer::<T, B, H, SelfM, CrossM>(
            tgt,
            memory,
            TransformerDecoderLayerParams {
                norm1_weight: &self.norm1.weight,
                norm1_bias: &self.norm1.bias,
                self_attn: MhaProjectionParams {
                    w_q: &self.self_attn.w_q,
                    b_q: self.self_attn.b_q.as_ref(),
                    w_k: &self.self_attn.w_k,
                    b_k: self.self_attn.b_k.as_ref(),
                    w_v: &self.self_attn.w_v,
                    b_v: self.self_attn.b_v.as_ref(),
                    w_o: &self.self_attn.w_o,
                    b_o: self.self_attn.b_o.as_ref(),
                },
                norm2_weight: &self.norm2.weight,
                norm2_bias: &self.norm2.bias,
                cross_attn: MhaProjectionParams {
                    w_q: &self.cross_attn.w_q,
                    b_q: self.cross_attn.b_q.as_ref(),
                    w_k: &self.cross_attn.w_k,
                    b_k: self.cross_attn.b_k.as_ref(),
                    w_v: &self.cross_attn.w_v,
                    b_v: self.cross_attn.b_v.as_ref(),
                    w_o: &self.cross_attn.w_o,
                    b_o: self.cross_attn.b_o.as_ref(),
                },
                norm3_weight: &self.norm3.weight,
                norm3_bias: &self.norm3.bias,
                ffn_w1: &self.ffn.linear1.weight,
                ffn_b1: self.ffn.linear1.bias.as_ref(),
                ffn_w2: &self.ffn.linear2.weight,
                ffn_b2: self.ffn.linear2.bias.as_ref(),
                self_attn_residual_dropout_p: self.dropout1.p,
                self_attn_residual_training: self.dropout1.is_training,
                cross_attn_residual_dropout_p: self.dropout2.p,
                cross_attn_residual_training: self.dropout2.is_training,
                ffn_hidden_dropout_p: self.ffn.dropout.p,
                ffn_hidden_training: self.ffn.dropout.is_training,
                ffn_residual_dropout_p: self.dropout3.p,
                ffn_residual_training: self.dropout3.is_training,
            },
        )
    }
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        SelfM: AttentionMask,
        CrossM: AttentionMask,
    > TransformerDecoderLayer<T, B, H, SelfM, CrossM>
{
    /// Collect all trainable decoder-layer parameters.
    pub fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.norm1.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.cross_attn.parameters());
        p.extend(self.norm3.parameters());
        p.extend(self.ffn.parameters());
        p
    }

    /// Collect trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("norm1", &self.norm1);
        parameters.extend(prefixed_parameters("self_attention", &self.self_attn));
        parameters.extend(prefixed_parameters("norm2", &self.norm2));
        parameters.extend(prefixed_parameters("cross_attention", &self.cross_attn));
        parameters.extend(prefixed_parameters("norm3", &self.norm3));
        parameters.extend(prefixed_parameters("feed_forward", &self.ffn));
        parameters
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
