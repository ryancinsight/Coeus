//! CPU linear algebra operations
//!
//! Linear algebra primitives optimized for CPU execution.
//! These operations handle matrix operations and decompositions.

pub mod matmul;
pub mod transpose;
pub mod decomposition;

// Re-export operations for convenience
pub use matmul::matmul_primitive;
pub use transpose::transpose_primitive;
pub use decomposition::*;