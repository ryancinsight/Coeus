// ── Attention module root ──

pub mod mask;
pub mod mha;
pub mod sdp;

// Re-export from coeus_autograd (the single authoritative source)
pub use coeus_autograd::{AttentionMask, CausalMask, NullMask};
pub use mha::MultiHeadAttention;
pub use sdp::ScaledDotProductAttention;
