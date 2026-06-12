// ── Transformer Encoder stack ──
//
// N independently-initialized encoder layers in a fixed-size array.
// `N` is a const generic: `[TransformerEncoderLayer; N]` is stack-allocated with
// no heap overhead for the container. Monomorphization emits one specialization
// per `(T, B, H, N, M)`.

use super::encoder_layer::TransformerEncoderLayer;
use crate::module::Module;
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
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        assert!(N > 0, "TransformerEncoder: N must be > 0");
        // `core::array::from_fn` calls the closure N times with indices 0..N.
        // Each call independently constructs a new layer with fresh parameters.
        let layers = core::array::from_fn(|_| {
            TransformerEncoderLayer::<T, B, H, M>::new(d_model, d_ff, dropout_p)
        });
        Self { layers }
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

    /// Forward through all N layers sequentially.
    ///
    /// Input/output shape: `[batch, seq, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.layers
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }
}
