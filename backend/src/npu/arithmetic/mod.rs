//! NPU arithmetic operations
//!
//! This module provides NPU-optimized implementations of basic arithmetic operations.

pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

// Re-export for convenience
pub use add::*;
pub use sub::*;
pub use mul::*;
pub use div::*;
