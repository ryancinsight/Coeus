// ── Attention module root ──

/// Attention mask types (causal, null).
pub mod mask;
/// Multi-head attention layer.
pub mod mha;
/// Scaled dot-product attention layer.
pub mod sdp;
mod validation;

// Re-export from coeus_autograd (the single authoritative source)
pub use coeus_autograd::{AttentionMask, CausalMask, NullMask};
pub use mha::{multi_head_attention_cross, MhaProjectionParams, MultiHeadAttention};
pub use sdp::ScaledDotProductAttention;
