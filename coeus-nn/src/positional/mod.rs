// ── Positional encoding module root ──

pub mod sinusoidal;
pub mod rope;

pub use sinusoidal::SinusoidalEncoding;
pub use rope::RotaryEmbedding;

