//! Compressed Sparse Row (CSR) storage module
//!
//! Provides CSR storage format optimized for row-wise access.
//!
//! ## Module Structure
//! - `core` - Struct definition and accessors
//! - `creation` - Constructors (empty, eye, from_dense)
//! - `conversion` - Format conversions (to_csc, to_coo, to_dense)
//! - `traits` - Trait implementations (Storage, AsAny)

mod conversion;
mod core;
mod creation;
mod traits;

pub use self::core::CsrStorage;
