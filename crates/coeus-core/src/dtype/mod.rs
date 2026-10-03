//! Scalar, Float, and Int trait hierarchies with impls for all numeric types,
//! and the operation tags dispatched over them.

mod complex;
mod float;
mod int;
mod reduction;
mod traits;

pub use eunomia::{Complex, CountRangeError, FloatElement, TryFromCount};
pub use reduction::{ClosedReduction, ReductionOp};
pub use traits::{BinaryOp, CpuUnaryDispatch, CpuUnaryOp, Float, FloatOps, Int, Scalar};
