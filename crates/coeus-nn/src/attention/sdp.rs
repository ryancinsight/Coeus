// ── Scaled Dot-Product Attention module ──

use crate::module::Module;
use coeus_autograd::{AttentionMask, Var};
use coeus_core::{Float, MoiraiBackend};
use std::marker::PhantomData;

/// Scaled dot-product attention layer.
///
/// `M` is a zero-sized mask strategy type sourced from `coeus_autograd`.
/// The active masking path is selected at compile time via `M::IS_CAUSAL`.
///
/// # Forward shape
/// - query: `[batch, seq_q, d_k]`, key: `[batch, seq_k, d_k]`, value: `[batch, seq_k, d_v]`
/// - output: `[batch, seq_q, d_v]`
#[derive(Clone, Copy)]
pub struct ScaledDotProductAttention<
    T: coeus_core::Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    M: AttentionMask = coeus_autograd::NullMask,
> {
    _marker: PhantomData<(T, B, M)>,
}

impl<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T> + Default, M: AttentionMask> Default
    for ScaledDotProductAttention<T, B, M>
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, M: AttentionMask>
    ScaledDotProductAttention<T, B, M>
{
    /// Create a new `ScaledDotProductAttention` layer (stateless, no parameters).
    pub fn new() -> Self {
        Self::default()
    }

    /// Forward pass with explicit Q, K, V.
    pub fn forward(
        &self,
        query: &Var<T, B>,
        key: &Var<T, B>,
        value: &Var<T, B>,
        key_padding_mask: Option<&Var<T, B>>,
        scale: T,
    ) -> Result<Var<T, B>, B::Error> {
        let (out, _attn_weights) =
            coeus_autograd::sdp_attention::<T, B, M>(query, key, value, key_padding_mask, scale)?;
        Ok(out)
    }
}

/// Convenience `Module` impl — self-attention: Q = K = V = input.
impl<T: Float, B: coeus_ops::BackendOps<T> + Default, M: AttentionMask> Module<T, B>
    for ScaledDotProductAttention<T, B, M>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        let d_k = input.tensor.shape()[2];
        let scale = T::one() / T::from_f64((d_k as f64).sqrt());
        self.forward(input, input, input, None, scale)
    }
}
