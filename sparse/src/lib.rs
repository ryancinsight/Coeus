//! # Coeus Sparse
//!
//! High-performance sparse numerical kernels and operations for the Coeus tensor library.
//! This crate provides operations for CSR, CSC, and COO sparse formats, separating
//! numerical logic from the foundational memory layouts in the `storage` crate.

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::*;
#[cfg(feature = "gpu")]
pub use gpu::*;

pub use storage::{CooStorage, CscStorage, CsrStorage, SparseFormat, Storage, StorageError};
pub type Result<T> = core::result::Result<T, StorageError>;
