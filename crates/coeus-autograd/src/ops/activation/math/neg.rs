//! Tracked element-wise negation.

use super::super::{unary_op, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;

/// ZST tag for negation autograd. Works for any `Scalar` (not just `Float`).
pub struct NegOp;
impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for NegOp {
    const OP_NAME: &'static str = "neg";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::neg(x, backend)
    }

    /// d/dx [−x] = −1, so grad_in = −grad_out.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::neg(grad_out, backend)
    }
}

/// Tracked element-wise negation.
#[must_use]
#[inline]
pub fn neg<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Var<T, B> {
    unary_op::<T, B, NegOp>(a)
}
