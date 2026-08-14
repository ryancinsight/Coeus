#![expect(
    clippy::module_inception,
    reason = "ratchet COEUS-LINT-1: transformer::transformer is the canonical composite module"
)]
// ── Transformer module root ──

/// Transformer decoder stack (N layers).
pub mod decoder;
/// Single transformer decoder layer (self-attn + cross-attn + FFN).
pub mod decoder_layer;
/// Transformer encoder stack (N layers).
pub mod encoder;
/// Single transformer encoder layer (self-attn + FFN).
pub mod encoder_layer;
/// Feed-forward sub-layer (two linear layers with GELU activation).
pub mod ffn;
mod normalization;
/// Full Seq2Seq Transformer model (encoder + decoder).
pub mod transformer;
mod validation;

pub use decoder::TransformerDecoder;
pub use decoder_layer::{
    transformer_decoder_layer, TransformerDecoderLayer, TransformerDecoderLayerParams,
};
pub use encoder::TransformerEncoder;
pub use encoder_layer::{
    transformer_encoder_layer, TransformerEncoderLayer, TransformerEncoderLayerParams,
};
pub use ffn::{feed_forward, FeedForward};
pub use transformer::Transformer;
