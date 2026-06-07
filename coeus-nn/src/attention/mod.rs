// ── Attention module root ──

pub mod mask;
pub mod sdp;
pub mod mha;

// Re-export from coeus_autograd (the single authoritative source)
pub use coeus_autograd::{AttentionMask, CausalMask, NullMask};
pub use sdp::ScaledDotProductAttention;
pub use mha::MultiHeadAttention;
