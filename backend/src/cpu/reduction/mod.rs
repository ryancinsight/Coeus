//! CPU reduction operation primitives
//!
//! Reduction operation primitives optimized for CPU execution.
//! These operations reduce tensors along specified dimensions.

pub mod sum;
pub mod mean;
pub mod max;

// Re-export operations for convenience
pub use sum::sum_primitive;
pub use mean::mean_primitive;
pub use max::{max_primitive, min_primitive};