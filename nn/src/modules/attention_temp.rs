//! Attention mechanisms for transformers (Legacy monolithic module)
//!
//! **DEPRECATED**: This module contains the original monolithic attention implementations.
//! New code should use the modular attention modules:
//! - `attention_config` for configuration structures
//! - `multihead_attention` for multi-head attention layers
//! - `causal_self_attention` for causal self-attention layers
//! - Legacy transformer components (Block, MLP, Transformer*) remain for compatibility
//!
//! ## Mathematical Foundation
//!
//! ### Self-Attention
//!
//! ```math
//! Attention(Q, K, V) = softmax(QK^T / √d_k)V
//! ```
//!
//! ### Multi-Head Attention
//!
//! ```math
//! MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O
//! head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
//! ```
//!
//! ### Causal Mask
//!
//! For causal (autoregressive) attention:
//!
//! ```math
//! mask[i,j] = 1 if j ≤ i else -∞
//! ```
//!
//! ## References
//!
//! - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
//! - [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};

// Re-export legacy monolithic implementations for backward compatibility
// These will be removed in a future version


