// ── Transformer Decoder stack ──

use super::decoder_layer::TransformerDecoderLayer;
use crate::module::{prefixed_parameters, Module};
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::{Float, MoiraiBackend};

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
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Result<Self, B::Error>
    where
        T: coeus_leto::RandomScalar,
    {
        let layers = (0..N)
            .map(|_| {
                TransformerDecoderLayer::<T, B, H, SelfM, CrossM>::new(
                    d_model,
                    d_ff,
                    dropout_p,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layers = layers.try_into().map_err(|_| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "transformer decoder layer array",
                reason: "layer count changed during fixed-array construction".to_owned(),
            })
        })?;
        Ok(Self { layers })
    }

    /// Forward decoder sequentially through the stack.
    ///
    /// Input/output shape: `[batch, seq_tgt, d_model]`.
    pub fn forward_decoder(
        &self,
        tgt: &Var<T, B>,
        memory: &Var<T, B>,
    ) -> Result<Var<T, B>, B::Error> {
        self.layers.iter().try_fold(tgt.clone(), |x, layer| {
            layer.forward_decoder(&x, memory)
        })
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

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| prefixed_parameters(&format!("layers.{index}"), layer))
            .collect()
    }

    /// Fallback forward without cross-attention.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
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
