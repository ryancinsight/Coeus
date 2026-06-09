// ── Positional encoding module root ──

pub mod rope;
pub mod sinusoidal;

pub use rope::RotaryEmbedding;
pub use sinusoidal::SinusoidalEncoding;
