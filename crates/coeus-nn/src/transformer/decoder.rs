// ── Transformer Decoder stack ──

use super::decoder_layer::TransformerDecoderLayer;
use crate::module::ModuleError;
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::MoiraiBackend;

/// Stack of N `TransformerDecoderLayer`s.
///
/// # Type parameters
/// - `H` — attention heads per layer (const generic)
/// - `N` — number of stacked decoder layers (const generic)
/// - `SelfM` — self-attention masking strategy ZST
/// - `CrossM` — cross-attention masking strategy ZST
pub struct TransformerDecoder<
    T: coeus_ops::AttentionScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    const N: usize = 6,
    SelfM: AttentionMask = CausalMask,
    CrossM: AttentionMask = NullMask,
> {
    /// Fixed-size array of decoder layers — stack-allocated container.
    pub layers: [TransformerDecoderLayer<T, B, H, SelfM, CrossM>; N],
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
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
    /// # Errors
    /// Returns an initialization error when `N == 0` or a layer cannot be
    /// initialized by the selected backend.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        T: coeus_leto::RandomScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        if N == 0 {
            return Err(crate::init::InitializationError::InvalidLayerCount { layers: N });
        }
        let layers = (0..N)
            .map(|_| {
                TransformerDecoderLayer::<T, B, H, SelfM, CrossM>::new(d_model, d_ff, dropout_p)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| crate::init::InitializationError::InvalidLayerCount { layers: N })?;
        Ok(Self { layers })
    }

    /// Forward decoder sequentially through the stack.
    ///
    /// Input/output shape: `[batch, seq_tgt, d_model]`.
    pub fn forward_decoder(
        &self,
        tgt: &Var<T, B>,
        memory: &Var<T, B>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward_decoder(&x, memory)?;
        }
        Ok(x)
    }
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        const N: usize,
        SelfM: AttentionMask,
        CrossM: AttentionMask,
    > TransformerDecoder<T, B, H, N, SelfM, CrossM>
{
    /// Collect all trainable decoder-stack parameters.
    pub fn parameters(&self) -> Vec<Var<T, B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Collect trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| {
                let prefix = format!("layers.{index}");
                layer
                    .named_parameters()
                    .into_iter()
                    .map(move |parameter| parameter.with_prefix(&prefix))
            })
            .collect()
    }
}

/// Manual Clone impl.
impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
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
