//! Compressed Sparse Column (CSC) storage module
//!
//! Provides CSC storage format optimized for column-wise access.

mod conversion;
mod core;
mod creation;
mod traits;

pub use self::core::CscStorage;
