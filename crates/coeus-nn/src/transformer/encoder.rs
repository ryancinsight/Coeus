// ── Transformer Encoder stack ──
//
// N independently-initialized encoder layers in a fixed-size array.
// `N` is a const generic: `[TransformerEncoderLayer; N]` is stack-allocated with
// no heap overhead for the container. Monomorphization emits one specialization
// per `(T, B, H, N, M)`.

use super::encoder_layer::TransformerEncoderLayer;
use crate::module::{prefixed_parameters, Module};
use coeus_autograd::{AttentionMask, Var};
use coeus_core::{Float, MoiraiBackend};

/// Stack of N `TransformerEncoderLayer`s.
///
/// # Type parameters
/// - `H` — attention heads per layer (const generic, uniform across layers)
/// - `N` — number of stacked encoder layers (const generic)
/// - `M` — masking strategy ZST
#[derive(Clone)]
pub struct TransformerEncoder<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    const N: usize = 6,
    M: AttentionMask = coeus_autograd::NullMask,
> {
    /// Fixed-size array of encoder layers — no heap allocation for the stack.
    pub layers: [TransformerEncoderLayer<T, B, H, M>; N],
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        const N: usize,
        M: AttentionMask,
    > TransformerEncoder<T, B, H, N, M>
where
    TransformerEncoderLayer<T, B, H, M>: Clone,
{
    /// Construct N independently-initialized encoder layers.
    ///
    /// # Panics
    /// Panics if `N == 0`.
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Result<Self, B::Error>
    where
        T: coeus_leto::RandomScalar,
    {
        let layers = (0..N)
            .map(|_| TransformerEncoderLayer::<T, B, H, M>::new(d_model, d_ff, dropout_p))
            .collect::<Result<Vec<_>, _>>()?;
        let layers = layers.try_into().map_err(|_| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "transformer encoder layer array",
                reason: "layer count changed during fixed-array construction".to_owned(),
            })
        })?;
        Ok(Self { layers })
    }

    /// Forward through all N layers sequentially with optional key padding mask.
    ///
    /// Input/output shape: `[batch, seq, d_model]`.
    /// `key_padding_mask` shape: `[batch, seq]` (or backend-supported broadcast form).
    pub fn forward_with_mask(
        &self,
        input: &Var<T, B>,
        key_padding_mask: Option<&Var<T, B>>,
    ) -> Result<Var<T, B>, B::Error> {
        self.layers.iter().try_fold(input.clone(), |x, layer| {
            layer.forward_with_mask(&x, key_padding_mask)
        })
    }
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        const N: usize,
        M: AttentionMask,
    > Module<T, B> for TransformerEncoder<T, B, H, N, M>
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

    /// Forward through all N layers sequentially.
    ///
    /// Input/output shape: `[batch, seq, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        self.forward_with_mask(input, None)
    }
}
