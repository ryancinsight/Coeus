//! CPU linear algebra operations
//!
//! Linear algebra primitives optimized for CPU execution.
//! These operations handle matrix operations and decompositions.

pub mod decomposition;
pub mod matmul;
pub mod matrix_exp;
pub mod transpose;

// Re-export operations for convenience
pub use decomposition::*;
pub use matmul::{matmul_primitive, gemm_primitive};
pub use matrix_exp::*;
pub use transpose::transpose_primitive;
