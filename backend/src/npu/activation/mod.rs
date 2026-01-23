//! NPU activation operations
//!
//! This module provides NPU-optimized implementations of activation functions.

pub mod relu;
pub mod sigmoid;
pub mod tanh;

// Re-export for convenience
pub use relu::*;
pub use sigmoid::*;
pub use tanh::*;
