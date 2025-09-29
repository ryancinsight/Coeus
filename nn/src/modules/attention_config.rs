//! Configuration for attention layers
//!
//! This module provides configuration structures for attention mechanisms
//! used in transformer architectures.

/// Configuration for attention layers
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_head: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Maximum sequence length
    pub block_size: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use causal masking
    pub causal: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            n_head: 12,
            n_embd: 768,
            block_size: 1024,
            dropout: 0.1,
            causal: true,
        }
    }
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(n_head: usize, n_embd: usize, block_size: usize, dropout: f64, causal: bool) -> Self {
        Self {
            n_head,
            n_embd,
            block_size,
            dropout,
            causal,
        }
    }
}


