// ── Transformer module root ──
#![allow(clippy::module_inception)]

pub mod decoder;
pub mod decoder_layer;
pub mod encoder;
pub mod encoder_layer;
pub mod ffn;
pub mod transformer;

pub use decoder::TransformerDecoder;
pub use decoder_layer::TransformerDecoderLayer;
pub use encoder::TransformerEncoder;
pub use encoder_layer::TransformerEncoderLayer;
pub use ffn::FeedForward;
pub use transformer::Transformer;
