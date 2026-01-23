//! Coordinate (COO) sparse storage module
//!
//! Provides COO storage format optimized for construction.

mod conversion;
mod core;
mod creation;
mod traits;

pub use self::core::CooStorage;
