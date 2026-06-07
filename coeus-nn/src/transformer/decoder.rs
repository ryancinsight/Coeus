// ── Transformer Decoder stack ──

use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::{Var, AttentionMask, CausalMask, NullMask};
use crate::module::Module;
use super::decoder_layer::TransformerDecoderLayer;

/// Stack of N `TransformerDecoderLayer`s.
///
/// # Type parameters
/// - `H` — attention heads per layer (const generic)
/// - `N` — number of stacked decoder layers (const generic)
/// - `SelfM` — self-attention masking strategy ZST
/// - `CrossM` — cross-attention masking strategy ZST
pub struct TransformerDecoder<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    const N: usize = 6,
    SelfM: AttentionMask = CausalMask,
    CrossM: AttentionMask = NullMask,
> {
    /// Fixed-size array of decoder layers — stack-allocated container.
    pub layers: [TransformerDecoderLayer<T, B, H, SelfM, CrossM>; N],
}

impl<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const H: usize,
    const N: usize,
    SelfM: AttentionMask,
    CrossM: AttentionMask,
> TransformerDecoder<T, B, H, N, SelfM, CrossM>
where
    TransformerDecoderLayer<T, B, H, SelfM, CrossM>: Clone,
{
    /// Construct N independently-initialized decoder layers.
    ///
    /// # Panics
    /// Panics if `N == 0`.
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self {
        assert!(N > 0, "TransformerDecoder: N must be > 0");
        let layers = core::array::from_fn(|_| {
            TransformerDecoderLayer::<T, B, H, SelfM, CrossM>::new(d_model, d_ff, dropout_p)
        });
        Self { layers }
    }

    /// Forward decoder sequentially through the stack.
    ///
    /// Input/output shape: `[batch, seq_tgt, d_model]`.
    pub fn forward_decoder(&self, tgt: &Var<T, B>, memory: &Var<T, B>) -> Var<T, B> {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward_decoder(&x, memory);
        }
        x
    }
}

impl<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const H: usize,
    const N: usize,
    SelfM: AttentionMask,
    CrossM: AttentionMask,
> Module<T, B> for TransformerDecoder<T, B, H, N, SelfM, CrossM>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Fallback forward without cross-attention.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward_decoder(input, input)
    }
}

/// Manual Clone impl.
impl<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const H: usize,
    const N: usize,
    SelfM: AttentionMask,
    CrossM: AttentionMask,
> Clone for TransformerDecoder<T, B, H, N, SelfM, CrossM>
where
    TransformerDecoderLayer<T, B, H, SelfM, CrossM>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}
