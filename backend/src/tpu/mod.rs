//! TPU (Tensor Processing Unit) backend implementation
//!
//! This module provides TPU-specific implementations of backend operations.
//! TPU operations are optimized for tensor computations and machine learning workloads.

pub mod arithmetic;
pub mod linear_algebra;
pub mod activation;
pub mod reduction;

// Re-export commonly used items
pub use arithmetic::*;
pub use linear_algebra::*;
pub use activation::*;
pub use reduction::*;
