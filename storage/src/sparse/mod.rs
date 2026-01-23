//! Sparse storage formats module
//!
//! Provides distinct sparse matrix storage types with consistent submodule structure:
//! - `csr/` - Compressed Sparse Row
//! - `csc/` - Compressed Sparse Column
//! - `coo/` - Coordinate format
//!
//! ## Submodule Structure
//! Each format contains:
//! - `core.rs` - Struct definition and accessors
//! - `creation.rs` - Constructors (empty, from_dense, etc.)
//! - `conversion.rs` - Format conversions (to_csr, to_csc, to_coo, to_dense)
//! - `traits.rs` - Trait implementations (Storage, AsAny)

pub mod coo;
pub mod csc;
pub mod csr;

pub use coo::CooStorage;
pub use csc::CscStorage;
pub use csr::CsrStorage;

/// Type alias for general sparse storage (defaults to CSR)
pub type SparseStorage<T> = CsrStorage<T>;

/// Sparse matrix format enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Compressed Sparse Row
    Csr,
    /// Compressed Sparse Column
    Csc,
    /// Coordinate format
    Coo,
}

impl Default for SparseFormat {
    fn default() -> Self {
        Self::Csr
    }
}
