//! CSR arithmetic operations
//!
//! This module provides arithmetic operations for CSR sparse matrices.

pub mod add;
pub mod mul;
pub mod matmul;

// Re-export main functions for convenience
pub use add::add;
pub use mul::mul;
pub use matmul::{matmul_sparse, matvec_mul, matmul_dense};
