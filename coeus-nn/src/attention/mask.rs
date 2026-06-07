// ── Attention mask ZST tags — re-exported from coeus-autograd ──
//
// The authoritative definitions live in `coeus_autograd::ops::nn::attention`.
// This module re-exports them so consumers can use `coeus_nn::CausalMask`
// without depending on `coeus_autograd` directly.

pub use coeus_autograd::{AttentionMask, CausalMask, NullMask};
