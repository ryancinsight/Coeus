// ── Dtype module ──
// Scalar, Float, and Int trait hierarchies with impls for all numeric types.

mod traits;
mod float;
mod int;
mod complex;

pub use traits::{Scalar, Float, Int};
pub use complex::Complex;
