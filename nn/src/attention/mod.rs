//! Attention mechanisms for neural networks.
//!
//! This module provides various attention mechanisms commonly used in transformer architectures:
//! - Multi-Head Attention (MHA)
//! - Sparse Attention
//! - Key-Value Cache (KV Cache)
//! - Quantized Attention variants
//! - Attention utilities and traits

pub mod multihead;
pub mod sparse;
pub mod utils;
pub mod kv_cache;

// Re-export commonly used attention types
pub use multihead::MultiHeadAttention;
pub use sparse::SparseAttention;
pub use utils::{AttentionDispatch, DenseAttention, SparseAttentionImpl, DenseStorageMarker, SparseStorageMarker};
pub use kv_cache::KVCache;

// Re-export quantized variants if feature is enabled
#[cfg(feature = "quantized")]
pub use kv_cache::{QuantizedKVCache, KVCacheCompressionStats};

// Re-export quantized variants if feature is enabled
#[cfg(feature = "quantized")]
pub use kv_cache::{QuantizedMultiHeadAttention, QuantizedSparseAttention};
