//! GPU reduction operation primitives (placeholder)

pub mod sum;
pub mod mean;
pub mod max;

pub use sum::sum_primitive;
pub use mean::mean_primitive;
pub use max::{max_primitive, min_primitive};