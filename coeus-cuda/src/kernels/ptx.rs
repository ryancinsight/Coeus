// ── Embedded PTX source for CUDA kernels ──

/// Embedded PTX source string compiled from the hand-written CUDA kernels.
pub const PTX_SOURCE: &str = include_str!("ptx.ptx");
