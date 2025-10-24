//! Attention mechanisms for neural networks.
//!
//! This module provides various attention mechanisms commonly used in transformer architectures:
//! - Multi-Head Attention (MHA)
//! - Sparse Attention
//! - Key-Value Cache (KV Cache)
//! - Quantized Attention variants
//! - Attention utilities and traits

pub mod kv_cache;
pub mod multihead;
pub mod sparse;
pub mod utils;

// Re-export commonly used attention types
pub use kv_cache::KVCache;
pub use multihead::MultiHeadAttention;
pub use sparse::{SparseAttention, SparseAttentionPattern};
pub use utils::{
    AttentionDispatch, DenseAttention, DenseStorageMarker, SparseAttentionImpl, SparseStorageMarker,
};

// Re-export quantized variants if feature is enabled
#[cfg(feature = "quantized")]
pub use kv_cache::{KVCacheCompressionStats, QuantizedKVCache};

// Re-export quantized variants if feature is enabled
#[cfg(feature = "quantized")]
pub use kv_cache::{QuantizedMultiHeadAttention, QuantizedSparseAttention};
