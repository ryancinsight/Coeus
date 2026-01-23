//! GPU backend implementation with hierarchical organization
//!
//! This module provides GPU backend operations organized in a hierarchical structure
//! to enable parity tracking with CPU, TPU, and NPU backends.
//!
//! ## Organization
//!
//! - `arithmetic/` - Basic arithmetic operations (add, sub, mul, div)
//! - `linear_algebra/` - Linear algebra operations (matmul, transpose, decomposition)
//! - `activation/` - Activation function primitives (relu, sigmoid, tanh)
//! - `reduction/` - Reduction operations (sum, mean, max)
//! - `sparse/` - Sparse matrix operations (SpMV, sparse arithmetic)
//! - `backend.rs` - GpuBackend implementing the Backend trait
//! - `dense_executor.rs` - GPU kernel executor for dense operations
//! - `traits.rs` - GPU type traits for zero-cost dispatch
//!
//! This structure mirrors the CPU backend organization for parity comparison.

pub mod arithmetic;
pub mod linear_algebra;
pub mod activation;
pub mod reduction;
pub mod sparse;
pub mod backend;
pub mod dense_executor;
pub mod traits;

// Re-export the main GPU backend, executor, and traits
pub use backend::GpuBackend;
pub use dense_executor::{GpuDenseExecutor, get_gpu_executor};
pub use traits::{GpuFloat, gpu_ops};