//! CPU backend implementation with hierarchical organization
//!
//! This module provides CPU backend operations organized in a hierarchical structure
//! to enable parity tracking across different operation categories.
//!
//! ## Organization
//!
//! - `arithmetic/` - Basic arithmetic operations (add, sub, mul, div)
//! - `linear_algebra/` - Linear algebra operations (matmul, transpose, decomposition)
//! - `activation/` - Activation function primitives (relu, sigmoid, tanh)
//! - `reduction/` - Reduction operations (sum, mean, max)
//!
//! This structure mirrors the organization across GPU, TPU, and NPU backends
//! to enable script-based parity comparison.

pub mod arithmetic;
pub mod linear_algebra;
pub mod activation;
pub mod reduction;
pub mod sparse_kernels;
pub mod backend;

// Re-export the main CPU backend
pub use backend::CpuBackend;