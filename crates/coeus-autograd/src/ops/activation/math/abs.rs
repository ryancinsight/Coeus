//! Tracked element-wise absolute value.

use super::super::{unary_op, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;

/// ZST tag for absolute-value autograd.
pub struct AbsOp;
impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AbsOp {
    const OP_NAME: &'static str = "abs";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::abs(x, backend)
    }

    /// d/dx |x| = sign(x).
    ///
    /// Computed as `abs(x) / x` where `x ≠ 0`.  At `x = 0` the result is 0
    /// (convention matches PyTorch/JAX).  Implemented via `abs(x)` then
    /// element-wise division using the backend's existing primitives — no new
    /// kernel required.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        // sign(x) = abs(x) / x; where x = 0 this produces NaN, masked to 0
        // by multiplying with ReluGrad(x) mask (0 at x≤0) combined with
        // ReluGrad(-x) (0 at x≥0). A cleaner and backend-portable approach:
        // use the stored forward output y = |x| and divide by x, then mul by grad_out.
        // Avoid division instability at x=0 by zeroing via the mask pattern:
        // sign = (x > 0) - (x < 0) encoded as abs(x)/x with zero at x=0 clamped.
        //
        // Implementation: sign(x) = neg(neg_mask) where:
        //   pos_mask = ReluGrad(x)      — 1 where x > 0, 0 elsewhere
        //   neg_mask = ReluGrad(-x_neg) — 1 where x < 0, 0 elsewhere
        // sign = pos_mask - neg_mask
        let pos_mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad)
            .expect("elementwise_unary");
        let x_neg = coeus_ops::neg(x, backend);
        let neg_mask = coeus_ops::elementwise_unary(&x_neg, backend, coeus_ops::UnaryOp::ReluGrad)
            .expect("elementwise_unary");
        let sign = coeus_ops::sub(&pos_mask, &neg_mask, backend);
        coeus_ops::mul(grad_out, &sign, backend)
    }
}

/// Tracked element-wise absolute value.
#[must_use]
#[inline]
pub fn abs<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Var<T, B> {
    unary_op::<T, B, AbsOp>(a)
}
