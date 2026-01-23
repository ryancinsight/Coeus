//! TPU linear algebra operations
//!
//! This module provides TPU-optimized implementations of linear algebra operations.

pub mod matmul;
pub mod transpose;
pub mod decomposition;

// Re-export for convenience
pub use matmul::*;
pub use transpose::*;
pub use decomposition::*;
