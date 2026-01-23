#![no_std]

//! # Coeus Sparse
//!
//! High-performance sparse numerical kernels and operations for the Coeus tensor library.
//!
//! ## Architecture
//!
//! This crate provides operations for sparse matrices using the unified CSR format.
//! CSC and COO are type aliases for backward compatibility, all operations use CSR internally.
//!
//! ```text
//! sparse/
//! ├── arithmetic/      - Arithmetic operations (add, sub, mul, div)
//! ├── reduction/       - Reduction operations (sum, max, min, mean)
//! ├── activation/      - Activation functions (relu, sigmoid, tanh, gelu)
//! ├── linear_algebra/  - Matrix multiplication operations
//! ├── creation/        - Sparse matrix creation functions
//! └── layout/          - Layout operations (transpose, reshape)
//! ```
//!
//! ## Trait Hierarchy
//!
//! - `SparseArithmetic` - Unified trait for basic arithmetic
//! - `SparseMatMul` - Matrix multiplication operations  
//! - `SparseReduce` - Reduction operations (sum, mean, max, min)
//! - `SparseActivation` - Activation functions
//! - `SparseLayout` - Layout transformations

extern crate alloc;

pub mod arithmetic;
pub mod activation;
pub mod creation;
pub mod layout;
pub mod linear_algebra;
pub mod reduction;

// Re-export core traits
pub use arithmetic::{
    SparseAdd, SparseArithmetic, SparseDiv, SparseElementWise, SparseMul,
    SparseOptimizerOps, SparseSub,
};
pub use activation::{SparseActivation, SparseRelu, SparseSigmoid, SparseTanh, SparseGelu};
pub use creation::SparseCreation;
pub use layout::{SparseLayout, SparseReshape, SparseTranspose};
pub use linear_algebra::SparseMatMul;
pub use reduction::{SparseReduce, SparseSum, SparseMax, SparseMin, SparseMean};

// Re-export storage types
pub use storage::{CooStorage, CscStorage, CsrStorage, SparseFormat, Storage, StorageError};

/// Result type for sparse operations
pub type Result<T> = core::result::Result<T, StorageError>;
