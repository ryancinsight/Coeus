//! Sparse arithmetic operations
//!
//! This module provides arithmetic operations for sparse matrices.
//! All operations use the unified CSR format for optimal performance.
//!
//! ## Architecture
//!
//! ```text
//! sparse/arithmetic/
//! ├── mod.rs        - Trait definitions and SparseArithmetic umbrella trait
//! ├── add.rs        - SparseAdd trait + CsrStorage impl
//! ├── sub.rs        - SparseSub trait + CsrStorage impl
//! ├── mul.rs        - SparseMul trait + CsrStorage impl
//! ├── div.rs        - SparseDiv trait + CsrStorage impl
//! ├── elementwise.rs- Element-wise operations
//! ├── reduction.rs  - Reduction operations (sum, max, min)
//! └── optimizer.rs  - Gradient optimizer operations
//! ```
//!
//! Note: matmul operations are in linear_algebra/ module

pub mod add;
pub mod div;
pub mod elementwise;
pub mod mul;
pub mod optimizer;
pub mod reduction;
pub mod sub;

// Re-export traits
pub use add::SparseAdd;
pub use div::SparseDiv;
pub use elementwise::SparseElementWise;
pub use mul::SparseMul;
pub use optimizer::SparseOptimizerOps;
pub use reduction::SparseReduce;
pub use sub::SparseSub;

use dtype::DataType;

/// Unified sparse arithmetic trait
///
/// Combines all basic arithmetic operations into a single trait.
/// This provides API parity with dense::DenseArithmetic.
pub trait SparseArithmetic<T: DataType>:
    SparseAdd<T> + SparseSub<T> + SparseMul<T> + SparseDiv<T>
{
}

// Blanket implementation for any type implementing all component traits
impl<T, S> SparseArithmetic<T> for S
where
    T: DataType,
    S: SparseAdd<T> + SparseSub<T> + SparseMul<T> + SparseDiv<T>,
{
}
