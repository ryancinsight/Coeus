// ── Positional encoding module root ──

/// Rotary Positional Embedding (RoPE).
pub mod rope;
/// Sinusoidal (non-learnable) positional encoding.
pub mod sinusoidal;

pub use rope::RotaryEmbedding;
pub use sinusoidal::SinusoidalEncoding;
