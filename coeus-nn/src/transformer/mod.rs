// ── Transformer module root ──
#![allow(clippy::module_inception)]

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
/// Full Seq2Seq Transformer model (encoder + decoder).
pub mod transformer;

pub use decoder::TransformerDecoder;
pub use decoder_layer::TransformerDecoderLayer;
pub use encoder::TransformerEncoder;
pub use encoder_layer::TransformerEncoderLayer;
pub use ffn::{feed_forward, FeedForward};
pub use transformer::Transformer;
