// ── Transformer Seq2Seq Model ──

use super::decoder::TransformerDecoder;
use super::encoder::TransformerEncoder;
use crate::module::Module;
use coeus_autograd::{AttentionMask, CausalMask, NullMask, Var};
use coeus_core::{Float, MoiraiBackend};

/// Full Sequence-to-Sequence Transformer model.
///
/// Composes a `TransformerEncoder` and `TransformerDecoder`.
pub struct Transformer<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    const NUM_ENC: usize = 6,
    const NUM_DEC: usize = 6,
    EncM: AttentionMask = NullMask,
    DecSelfM: AttentionMask = CausalMask,
    DecCrossM: AttentionMask = NullMask,
> {
    pub encoder: TransformerEncoder<T, B, H, NUM_ENC, EncM>,
    pub decoder: TransformerDecoder<T, B, H, NUM_DEC, DecSelfM, DecCrossM>,
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
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
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        Self {
            encoder: TransformerEncoder::new(d_model, d_ff, dropout_p),
            decoder: TransformerDecoder::new(d_model, d_ff, dropout_p),
        }
    }

    /// Complete Seq2Seq forward pass.
    ///
    /// - `src`: input sequence `[batch, seq_src, d_model]`
    /// - `tgt`: target sequence `[batch, seq_tgt, d_model]`
    ///
    /// Returns decoded output `[batch, seq_tgt, d_model]`.
    pub fn forward_seq2seq(&self, src: &Var<T, B>, tgt: &Var<T, B>) -> Var<T, B> {
        let memory = self.encoder.forward_with_mask(src, None);
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
    ) -> Var<T, B> {
        let memory = self.encoder.forward_with_mask(src, src_key_padding_mask);
        self.decoder.forward_decoder(tgt, &memory)
    }
}

impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
        const H: usize,
        const NUM_ENC: usize,
        const NUM_DEC: usize,
        EncM: AttentionMask,
        DecSelfM: AttentionMask,
        DecCrossM: AttentionMask,
    > Module<T, B> for Transformer<T, B, H, NUM_ENC, NUM_DEC, EncM, DecSelfM, DecCrossM>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.encoder.parameters();
        p.extend(self.decoder.parameters());
        p
    }

    /// Fallback forward routing to `forward_seq2seq(input, input)`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward_seq2seq(input, input)
    }

    fn train(&mut self, mode: bool) {
        self.encoder.train(mode);
        self.decoder.train(mode);
    }
}

/// Manual Clone impl.
impl<
        T: Float,
        B: coeus_ops::BackendOps<T> + Default,
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
