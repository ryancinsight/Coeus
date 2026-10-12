//! CPU dispatch trait for unary operations.

use super::{op_tags::CpuUnaryOp, private};

/// CPU dispatch trait for unary operations.
///
/// Implemented for all `Scalar` types that support CPU-side unary kernels.
pub trait CpuUnaryDispatch: private::Sealed {
    /// Evaluate a unary operation on a single element.
    fn eval_unary(op: CpuUnaryOp, x: Self) -> Self;
}
