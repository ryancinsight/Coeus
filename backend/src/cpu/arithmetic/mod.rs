//! CPU arithmetic operations
//!
//! Basic element-wise arithmetic operations optimized for CPU execution.
//! These operations serve as the foundation for higher-level tensor operations.

pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

// Re-export operations for convenience
pub use add::add_primitive;
pub use sub::sub_primitive;
pub use mul::mul_primitive;
pub use div::div_primitive;