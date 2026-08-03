// ── Transformer Seq2Seq Model ──

use super::decoder::TransformerDecoder;
use super::encoder::TransformerEncoder;
use crate::module::{prefixed_parameters, Module, ModuleError};
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::MoiraiBackend;

/// Full Sequence-to-Sequence Transformer model.
///
/// Composes a `TransformerEncoder` and `TransformerDecoder`.
pub struct Transformer<
    T: coeus_ops::AttentionScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    const NUM_ENC: usize = 6,
    const NUM_DEC: usize = 6,
    EncM: AttentionMask = NullMask,
    DecSelfM: AttentionMask = CausalMask,
    DecCrossM: AttentionMask = NullMask,
> {
    /// Encoder stack processing the source sequence.
    pub encoder: TransformerEncoder<T, B, H, NUM_ENC, EncM>,
    /// Decoder stack producing the target sequence.
    pub decoder: TransformerDecoder<T, B, H, NUM_DEC, DecSelfM, DecCrossM>,
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        const NUM_ENC: usize,
        const NUM_DEC: usize,
        EncM: AttentionMask,
        DecSelfM: AttentionMask,
        DecCrossM: AttentionMask,
    > Transformer<T, B, H, NUM_ENC, NUM_DEC, EncM, DecSelfM, DecCrossM>
where
    TransformerEncoder<T, B, H, NUM_ENC, EncM>: Clone,
    TransformerDecoder<T, B, H, NUM_DEC, DecSelfM, DecCrossM>: Clone,
{
    /// Construct a new Seq2Seq Transformer model.
    ///
    /// # Errors
    ///
    /// Returns a typed error when either stack has no layers, attention
    /// dimensions are invalid, or provider initialization fails.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        T: coeus_leto::RandomScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        Ok(Self {
            encoder: TransformerEncoder::new(d_model, d_ff, dropout_p)?,
            decoder: TransformerDecoder::new(d_model, d_ff, dropout_p)?,
        })
    }

    /// Complete Seq2Seq forward pass.
    ///
    /// - `src`: input sequence `[batch, seq_src, d_model]`
    /// - `tgt`: target sequence `[batch, seq_tgt, d_model]`
    ///
    /// Returns decoded output `[batch, seq_tgt, d_model]`.
    pub fn forward_seq2seq(
        &self,
        src: &Var<T, B>,
        tgt: &Var<T, B>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let memory = self.encoder.forward_with_mask(src, None)?;
        self.decoder.forward_decoder(tgt, &memory)
    }

    /// Complete Seq2Seq forward pass with optional source key-padding mask.
    ///
    /// - `src_key_padding_mask`: mask for encoder self-attention keys, typically shape
    ///   `[batch, seq_src]` with non-zero for keep and zero for padded tokens.
    pub fn forward_seq2seq_with_src_mask(
        &self,
        src: &Var<T, B>,
        tgt: &Var<T, B>,
        src_key_padding_mask: Option<&Var<T, B>>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let memory = self.encoder.forward_with_mask(src, src_key_padding_mask)?;
        self.decoder.forward_decoder(tgt, &memory)
    }
}

impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        const NUM_ENC: usize,
        const NUM_DEC: usize,
        EncM: AttentionMask,
        DecSelfM: AttentionMask,
        DecCrossM: AttentionMask,
    > Transformer<T, B, H, NUM_ENC, NUM_DEC, EncM, DecSelfM, DecCrossM>
{
    /// Collect all trainable encoder and decoder parameters.
    pub fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.encoder.parameters();
        p.extend(self.decoder.parameters());
        p
    }

    /// Collect trainable parameters with stable encoder/decoder prefixes.
    pub fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("encoder", &self.encoder);
        parameters.extend(
            self.decoder
                .named_parameters()
                .into_iter()
                .map(|parameter| parameter.with_prefix("decoder")),
        );
        parameters
    }
}

/// Manual Clone impl.
impl<
        T: coeus_ops::AttentionScalar,
        B: coeus_ops::BackendOps<T> + coeus_ops::AttentionOps<T> + Default,
        const H: usize,
        const NUM_ENC: usize,
        const NUM_DEC: usize,
        EncM: AttentionMask,
        DecSelfM: AttentionMask,
        DecCrossM: AttentionMask,
    > Clone for Transformer<T, B, H, NUM_ENC, NUM_DEC, EncM, DecSelfM, DecCrossM>
where
    TransformerEncoder<T, B, H, NUM_ENC, EncM>: Clone,
    TransformerDecoder<T, B, H, NUM_DEC, DecSelfM, DecCrossM>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
        }
    }
}
