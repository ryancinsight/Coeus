use super::CpuUnaryOp;

/// CPU dispatch trait for unary operations.
///
/// Implemented for all `Scalar` types that support CPU-side unary kernels.
pub trait CpuUnaryDispatch: super::private::Sealed {
    /// Evaluate a unary operation on a single element.
    fn eval_unary(op: CpuUnaryOp, x: Self) -> Self;
}
