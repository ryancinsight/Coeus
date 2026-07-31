// ── Transformer Encoder Layer (Pre-LayerNorm) ──

use super::ffn::{feed_forward_with_training, FeedForward};
use crate::attention::{
    multi_head_attention_cross, MhaProjectionParams, MultiHeadAttention, NullMask,
};
use crate::dropout::Dropout;
use crate::module::{prefixed_parameters, Module, ModuleError};
use crate::normalization::LayerNorm;
use coeus_autograd::{AttentionMask, Var};
use coeus_core::MoiraiBackend;

/// Borrowed parameters for functional transformer encoder-layer execution.
pub struct TransformerEncoderLayerParams<
    'a,
    T: coeus_ops::AttentionScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
> {
    /// LayerNorm1 gamma `[d_model]`.
    pub norm1_weight: &'a Var<T, B>,
    /// LayerNorm1 beta `[d_model]`.
    pub norm1_bias: &'a Var<T, B>,
    /// Self-attention projections.
    pub self_attn: MhaProjectionParams<'a, T, B>,
    /// LayerNorm2 gamma `[d_model]`.
    pub norm2_weight: &'a Var<T, B>,
    /// LayerNorm2 beta `[d_model]`.
    pub norm2_bias: &'a Var<T, B>,
    /// FFN linear1 weight `[d_ff, d_model]`.
    pub ffn_w1: &'a Var<T, B>,
    /// FFN linear1 bias `[d_ff]`.
    pub ffn_b1: Option<&'a Var<T, B>>,
    /// FFN linear2 weight `[d_model, d_ff]`.
    pub ffn_w2: &'a Var<T, B>,
    /// FFN linear2 bias `[d_model]`.
    pub ffn_b2: Option<&'a Var<T, B>>,
    /// Dropout probability after self-attention.
    pub attn_residual_dropout_p: f64,
    /// Training mode for dropout after self-attention.
    pub attn_residual_training: bool,
    /// Dropout probability inside FFN hidden projection.
    pub ffn_hidden_dropout_p: f64,
    /// Training mode for FFN hidden dropout.
    pub ffn_hidden_training: bool,
    /// Dropout probability after FFN residual branch.
    pub ffn_residual_dropout_p: f64,
    /// Training mode for dropout after FFN.
    pub ffn_residual_training: bool,
}

fn apply_dropout<T: coeus_ops::AttentionScalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    p: f64,
    is_training: bool,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    let mut dropout = Dropout::new(p);
    dropout.set_training(is_training);
    dropout.forward(x)
}

/// Functional (stateless) TransformerEncoderLayer forward.
///
/// Computes pre-LN encoder sublayers:
/// `x1 = x + Dropout(SelfAttn(LN1(x)))`
/// `x2 = x1 + Dropout(Linear2(Dropout(GELU(Linear1(LN2(x1))))))`.
pub fn transformer_encoder_layer<
    T: coeus_ops::AttentionScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
    const H: usize,
    M: AttentionMask,
>(
    input: &Var<T, B>,
    key_padding_mask: Option<&Var<T, B>>,
    params: TransformerEncoderLayerParams<'_, T, B>,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    const MODULE: &str = "TransformerEncoderLayer";
    let [_, _, d_model] = super::validation::rank_three(MODULE, input)?;
    super::validation::affine_vector(MODULE, "norm1 weight", params.norm1_weight, d_model)?;
    super::validation::affine_vector(MODULE, "norm1 bias", params.norm1_bias, d_model)?;
    super::validation::affine_vector(MODULE, "norm2 weight", params.norm2_weight, d_model)?;
    super::validation::affine_vector(MODULE, "norm2 bias", params.norm2_bias, d_model)?;
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
        input,
        params.norm1_weight,
        params.norm1_bias,
        1e-5,
    )?;
    let attn_out = multi_head_attention_cross::<T, B, H, M>(
        &normed1,
        &normed1,
        &normed1,
        params.self_attn,
        key_padding_mask,
    )?;
    let dropped1 = apply_dropout(
        &attn_out,
        params.attn_residual_dropout_p,
        params.attn_residual_training,
    )?;
    let x = coeus_autograd::add(input, &dropped1);

    let normed2 = super::normalization::layer_norm_three_dimensional(
        MODULE,
        &x,
        params.norm2_weight,
        params.norm2_bias,
        1e-5,
    )?;
    let ff_out = feed_forward_with_training(
        &normed2,
        params.ffn_w1,
        params.ffn_b1,
        params.ffn_w2,
        params.ffn_b2,
        params.ffn_hidden_dropout_p,
        params.ffn_hidden_training,
    )?;
    let dropped2 = apply_dropout(
        &ff_out,
        params.ffn_residual_dropout_p,
        params.ffn_residual_training,
    )?;
    Ok(coeus_autograd::add(&x, &dropped2))
}

/// Single Transformer encoder layer.
///
/// Pre-LayerNorm sub-layer composition:
/// ```text
///   x₁ = x + Dropout(SelfAttn(LayerNorm(x)))
///   x₂ = x₁ + Dropout(FFN(LayerNorm(x₁)))
/// ```
pub struct TransformerEncoderLayer<
    T: coeus_ops::AttentionScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    M: AttentionMask = NullMask,
> {
    /// Pre-LayerNorm before self-attention.
    pub norm1: LayerNorm<T, B>,
    /// Self-attention sub-layer.
    pub self_attn: MultiHeadAttention<T, B, H, M>,
    /// Dropout applied after self-attention.
    pub dropout1: Dropout,
    /// Pre-LayerNorm before FFN.
    pub norm2: LayerNorm<T, B>,
    /// Feed-forward sub-layer.
    pub ffn: FeedForward<T, B>,
    /// Dropout applied after FFN.
    pub dropout2: Dropout,
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        M: AttentionMask,
    > TransformerEncoderLayer<T, B, H, M>
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

    /// Pre-LayerNorm forward with optional key padding mask.
    ///
    /// Input/output shape: `[batch, seq, d_model]`.
    /// `key_padding_mask` shape: `[batch, seq]` (or backend-supported broadcast form).
    pub fn forward_with_mask(
        &self,
        input: &Var<T, B>,
        key_padding_mask: Option<&Var<T, B>>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        transformer_encoder_layer::<T, B, H, M>(
            input,
            key_padding_mask,
            TransformerEncoderLayerParams {
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
                ffn_w1: &self.ffn.linear1.weight,
                ffn_b1: self.ffn.linear1.bias.as_ref(),
                ffn_w2: &self.ffn.linear2.weight,
                ffn_b2: self.ffn.linear2.bias.as_ref(),
                attn_residual_dropout_p: self.dropout1.p,
                attn_residual_training: self.dropout1.is_training,
                ffn_hidden_dropout_p: self.ffn.dropout.p,
                ffn_hidden_training: self.ffn.dropout.is_training,
                ffn_residual_dropout_p: self.dropout2.p,
                ffn_residual_training: self.dropout2.is_training,
            },
        )
    }
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        M: AttentionMask,
    > Module<T, B> for TransformerEncoderLayer<T, B, H, M>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.norm1.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.ffn.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("norm1", &self.norm1);
        parameters.extend(prefixed_parameters("self_attention", &self.self_attn));
        parameters.extend(prefixed_parameters("norm2", &self.norm2));
        parameters.extend(prefixed_parameters("feed_forward", &self.ffn));
        parameters
    }

    /// Pre-LayerNorm forward. Input/output: `[batch, seq, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        self.forward_with_mask(input, None)
    }
}

/// Manual Clone impl.
impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        M: AttentionMask,
    > Clone for TransformerEncoderLayer<T, B, H, M>
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
